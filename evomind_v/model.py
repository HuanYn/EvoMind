"""MiniMind-V with strict multi-view insertion and optional encoder-output cache.

Parameter names match the official model. The language transformer, head, loss,
RoPE, and KV handling below preserve the upstream forward computation; only the
vision encoding/insertion boundary changes.
"""

from collections.abc import Mapping

import torch
from torch.nn import functional as F
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from model.model_minimind import MiniMindForCausalLM, MOEFeedForward, precompute_freqs_cis
from model.model_vlm import MiniMindVLM, MMVisionProjector, VLMConfig


class EvoMindVLM(MiniMindVLM):
    def __init__(self, config=None, vision_model_path="./model/siglip2-base-p32-256-ve",
                 *, vision_encoder=None, processor=None, load_vision_encoder=True):
        config = config or VLMConfig()
        if config.image_token_len != 64:
            raise ValueError("EvoMind-V requires exactly 64 image tokens per view")
        if len(config.image_ids) != 1:
            raise ValueError("config.image_ids must contain exactly one image marker token ID")
        MiniMindForCausalLM.__init__(self, config)
        if vision_encoder is not None:
            self.vision_encoder, self.processor = vision_encoder, processor
        elif load_vision_encoder:
            self.vision_encoder, self.processor = self.get_vision_model(vision_model_path)
        else:
            self.vision_encoder, self.processor = None, processor
        if self.vision_encoder is not None:
            self.vision_encoder.requires_grad_(False)
            self.vision_encoder.float()
            self.vision_encoder.eval()
        self.vision_proj = MMVisionProjector(
            config.image_hidden_size, config.hidden_size, target_tokens=config.image_token_len
        )

    def train(self, mode=True):
        super().train(mode)
        # Module.train recursively toggles children, including frozen modules.
        if self.vision_encoder is not None:
            self.vision_encoder.eval()
        return self

    def _assert_frozen_encoder(self):
        if self.vision_encoder is not None and any(p.requires_grad for p in self.vision_encoder.parameters()):
            raise ValueError("vision feature caching/encoding requires a frozen vision encoder; disable encoder training")

    def _validate_features(self, features, batch_size):
        if not isinstance(features, torch.Tensor) or features.ndim != 4:
            raise ValueError("vision_features must have shape [batch, views, 64, hidden_size]")
        if features.shape[0] != batch_size or features.shape[1] not in (1, 5):
            raise ValueError("vision_features batch must match input_ids and views must be 1 or 5")
        if tuple(features.shape[2:]) != (64, self.config.image_hidden_size):
            raise ValueError(f"vision_features trailing shape must be (64, {self.config.image_hidden_size})")
        if not features.is_floating_point():
            raise ValueError("vision_features must be floating point")
        if features.requires_grad:
            raise ValueError("vision_features must be detached frozen encoder outputs")

    def encode_images(self, pixel_values):
        """Return frozen encoder outputs [B,V,64,D], never run the projector.

        Input is a tensor [B,V,C,H,W] (or upstream [B,V,1,C,H,W])
        or a processor mapping with that shape. [B,C,H,W] means one view.
        """
        self._assert_frozen_encoder()
        if self.vision_encoder is None:
            raise ValueError("pixel_values require a locally loaded vision encoder")
        inputs = dict(pixel_values) if isinstance(pixel_values, Mapping) else {"pixel_values": pixel_values}
        if "pixel_values" not in inputs:
            raise ValueError("processor inputs must include pixel_values")
        pixels = inputs["pixel_values"]
        if pixels.ndim == 6 and pixels.shape[2] == 1:
            inputs["pixel_values"] = pixels = pixels.squeeze(2)
        if pixels.ndim == 4:
            batch, views = pixels.shape[0], 1
            flattened = inputs
        elif pixels.ndim == 5:
            batch, views = pixels.shape[:2]
            flattened = {}
            for name, value in inputs.items():
                if value.ndim < 2 or tuple(value.shape[:2]) != (batch, views):
                    raise ValueError(f"processor tensor {name!r} must share [batch, views] axes")
                flattened[name] = value.flatten(0, 1)
        else:
            raise ValueError("pixel_values must have shape [B,V,C,H,W] or [B,C,H,W]")
        if views not in (1, 5):
            raise ValueError("expected one global view or five global/local views")
        self.vision_encoder.eval()
        parameter = next(self.vision_encoder.parameters(), None)
        if parameter is not None and parameter.dtype != torch.float32:
            raise ValueError("keep the frozen vision encoder in float32 so live and cached features share a compute dtype")
        if parameter is not None:
            flattened = {
                name: value.to(device=parameter.device, dtype=parameter.dtype if value.is_floating_point() else value.dtype)
                for name, value in flattened.items()
            }
        # no_grad (not inference_mode): these outputs must be usable by the
        # trainable projector's autograd backward pass.
        # Training autocast must not change encoder precision between live
        # training and offline preparation. Only the trainable model is AMP'd.
        device_type = parameter.device.type if parameter is not None else pixels.device.type
        with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
            features = self.vision_encoder(**flattened).last_hidden_state
        if tuple(features.shape) != (batch * views, 64, self.config.image_hidden_size):
            raise ValueError(f"vision encoder returned incompatible shape {tuple(features.shape)}")
        return features.reshape(batch, views, 64, self.config.image_hidden_size)

    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=None):
        if vision_tensors is None:
            return h
        if vision_tensors.ndim == 3:
            vision_tensors = vision_tensors.unsqueeze(1)
        marker = self.config.image_ids[0]
        output = []
        for batch_idx, sequence in enumerate(tokens.detach().cpu().tolist()):
            spans, index = [], 0
            while index < len(sequence):
                if sequence[index] != marker:
                    index += 1
                    continue
                start = index
                while index < len(sequence) and sequence[index] == marker:
                    index += 1
                spans.append((start, index))
            if len(spans) != vision_tensors.shape[1] or any(end - start != 64 for start, end in spans):
                raise ValueError(
                    f"sample {batch_idx}: expected {vision_tensors.shape[1]} separate image placeholder "
                    f"segments of exactly 64 tokens, found lengths {[end - start for start, end in spans]}"
                )
            pieces, cursor = [], 0
            for view_idx, (start, end) in enumerate(spans):
                pieces.extend((h[batch_idx, cursor:start], vision_tensors[batch_idx, view_idx].to(h.dtype)))
                cursor = end
            pieces.append(h[batch_idx, cursor:])
            output.append(torch.cat(pieces, dim=0))
        return torch.stack(output)

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                use_cache=False, logits_to_keep=0, labels=None,
                pixel_values=None, vision_features=None, **args):
        if input_ids is None or input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if pixel_values is not None and vision_features is not None:
            raise ValueError("supply either pixel_values or vision_features, not both")
        if vision_features is not None:
            self._assert_frozen_encoder()
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))
        if start_pos == 0:
            if vision_features is None and pixel_values is not None:
                vision_features = self.encode_images(pixel_values)
            if vision_features is not None:
                self._validate_features(vision_features, batch_size)
                parameter = next(self.vision_proj.parameters())
                features = vision_features.to(device=parameter.device, dtype=parameter.dtype)
                vision_tensors = self.vision_proj(features)
                hidden_states = self.count_vision_proj(input_ids, hidden_states, vision_tensors)
            elif bool((input_ids == self.config.image_ids[0]).any()):
                raise ValueError("image placeholders require pixel_values or vision_features")

        # The rest matches official MiniMindVLM.forward, including auxiliary
        # zero gradients for DDP and prefill/decode RoPE and KV-cache handling.
        if self.model.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim, end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling,
            )
            self.model.freqs_cos, self.model.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length],
        )
        presents = []
        for layer, past_key_value in zip(self.model.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states, position_embeddings, past_key_value=past_key_value,
                use_cache=use_cache, attention_mask=attention_mask,
            )
            presents.append(present)
        hidden_states = self.model.norm(hidden_states)
        aux_loss = sum([layer.mlp.aux_loss for layer in self.model.layers if isinstance(layer.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        aux_loss = aux_loss + sum(p.sum() for p in self.vision_proj.parameters()) * 0
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(
            loss=loss, aux_loss=aux_loss, logits=logits,
            past_key_values=presents, hidden_states=hidden_states,
        )

    def generate(self, *args, num_return_sequences=1, **kwargs):
        if num_return_sequences > 1 and kwargs.get("vision_features") is not None:
            features = kwargs["vision_features"]
            kwargs["vision_features"] = features.repeat(num_return_sequences, *([1] * (features.ndim - 1)))
        return super().generate(*args, num_return_sequences=num_return_sequences, **kwargs)
