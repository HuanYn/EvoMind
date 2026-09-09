"""CPU-only regression tests; no pretrained weights or network are used.

Run from vision/: python -m unittest discover -s test -p test_views_cache.py -v
"""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import torch
from torch import nn

from evomind_v.cache import VisionFeatureCache, fingerprint_vision
from evomind_v.model import EvoMindVLM, VLMConfig
from evomind_v.views import make_image_views, num_views, preprocess_views
from model.model_vlm import MiniMindVLM


class FakeProcessor:
    def __init__(self, size=8):
        self.size = size

    def to_dict(self):
        return {"size": self.size, "resample": "bilinear", "do_normalize": False}

    def __call__(self, images, return_tensors):
        if not isinstance(images, list):
            images = [images]
        values = [
            torch.tensor(list(image.resize((8, 8), Image.Resampling.BILINEAR).getdata()), dtype=torch.float32)
            .reshape(8, 8, 3).permute(2, 0, 1) / 255
            for image in images
        ]
        return {"pixel_values": torch.stack(values)}


class FakeEncoder(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.5, 1.5, hidden_size))
        self.dropout = nn.Dropout(0.7)
        self.calls = 0

    def forward(self, pixel_values):
        self.calls += 1
        patches = torch.nn.functional.adaptive_avg_pool2d(pixel_values, (8, 8))
        patches = patches.mean(dim=1).flatten(1).unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=self.dropout(patches @ self.weight.unsqueeze(0)))


def config():
    return VLMConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=48, vocab_size=32,
        max_position_embeddings=512, image_hidden_size=8,
        image_token_len=64, image_ids=[12], dropout=0.0, flash_attn=False,
    )


def token_batch(views=5, batch_size=1):
    sequence = [1, 3]
    for _ in range(views):
        sequence += [12] * 64 + [4]
    sequence += [5, 6, 2]
    return torch.tensor([sequence] * batch_size)


class ViewsTests(unittest.TestCase):
    def test_quadrants_order_rgb_and_odd_size(self):
        image = Image.new("RGBA", (5, 3))
        for y in range(3):
            for x in range(5):
                image.putpixel((x, y), (x * 30, y * 60, 10, 180))
        views = make_image_views(image, "multi")
        self.assertEqual([view.size for view in views], [(5, 3), (2, 1), (3, 1), (2, 2), (3, 2)])
        self.assertTrue(all(view.mode == "RGB" for view in views))
        self.assertEqual([view.getpixel((0, 0)) for view in views[1:]],
                         [(0, 0, 10), (60, 0, 10), (0, 60, 10), (60, 60, 10)])
        self.assertEqual([v.tobytes() for v in views], [v.tobytes() for v in make_image_views(image, "multi")])

    def test_one_pixel_grayscale_and_single(self):
        for size in [(1, 1), (1, 3), (3, 1)]:
            image = Image.new("L", size, 50)
            views = make_image_views(image, "multi")
            self.assertEqual(len(views), 5)
            self.assertTrue(all(view.width > 0 and view.height > 0 for view in views))
            self.assertEqual(views[0].getpixel((0, 0)), (50, 50, 50))
            self.assertEqual(len(make_image_views(image, "single")), 1)
        with self.assertRaises(ValueError):
            num_views("random")

    def test_preprocessor_has_view_axis(self):
        values = preprocess_views(Image.new("RGB", (7, 5), "red"), FakeProcessor(), "multi")
        self.assertEqual(tuple(values["pixel_values"].shape), (5, 3, 8, 8))


class CacheTests(unittest.TestCase):
    def test_roundtrip_raw_hash_dtype_and_namespaces(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = VisionFeatureCache(directory, "encoder-a", "multi")
            key = cache.key_for(b"original image bytes")
            self.assertIsNone(cache.load(key))
            features = torch.randn(5, 64, 768)
            path = cache.save(key, features)
            self.assertEqual(path, cache.path_for(key))
            torch.testing.assert_close(cache.load(key), features, rtol=0, atol=0)
            for kwargs in [{"fingerprint": "encoder-b"}, {"mode": "single"}, {"dtype": "float16"}]:
                arguments = {"fingerprint": "encoder-a", "mode": "multi", **kwargs}
                self.assertNotEqual(key, VisionFeatureCache(directory, **arguments).key_for(b"original image bytes"))
            self.assertNotEqual(key, cache.key_for(b"changed image bytes"))
            half = VisionFeatureCache(directory, "encoder-a", "multi", dtype="float16")
            half_key = half.key_for(b"original image bytes")
            half.save(half_key, features)
            self.assertEqual(half.load(half_key).dtype, torch.float16)
            self.assertEqual(half.load(half_key).device.type, "cpu")
            torch.testing.assert_close(half.load(half_key), features.half(), rtol=0, atol=0)
            self.assertFalse(list(Path(directory).rglob("*.tmp")))

    def test_file_key_uses_bytes_not_path_or_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpg"
            path.write_bytes(b"same bytes")
            cache = VisionFeatureCache(directory, "encoder-a")
            self.assertEqual(cache.key_for(path), cache.key_for(b"same bytes"))
            previous = cache.key_for(path)
            path.write_bytes(b"edit bytes")
            self.assertNotEqual(previous, cache.key_for(path))

    def test_weight_and_preprocessor_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model.safetensors"
            weights.write_bytes(b"fake-weights-1")
            processor = FakeProcessor()
            first = fingerprint_vision(directory, processor)
            self.assertEqual(first, fingerprint_vision(directory, processor))
            weights.write_bytes(b"fake-weights-2")
            second = fingerprint_vision(directory, processor)
            self.assertNotEqual(first, second)
            self.assertNotEqual(second, fingerprint_vision(directory, FakeProcessor(size=16)))

    def test_reject_shape_graph_overflow_traversal_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = VisionFeatureCache(directory, "encoder-a", hidden_size=8)
            key = cache.key_for(b"image")
            with self.assertRaises(ValueError):
                cache.save(key, torch.zeros(1, 63, 8))
            with self.assertRaises(ValueError):
                cache.save(key, torch.zeros(1, 64, 8, requires_grad=True))
            with self.assertRaises(ValueError):
                cache.path_for("../outside")
            half = VisionFeatureCache(directory, "encoder-a", dtype="float16", hidden_size=8)
            with self.assertRaises(ValueError):
                half.save(half.key_for(b"image"), torch.full((1, 64, 8), 1e10))
            path = cache.save(key, torch.zeros(1, 64, 8))
            payload = torch.load(path, weights_only=True)
            payload["metadata"]["fingerprint"] = "wrong"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "metadata/fingerprint"):
                cache.load(key)
            path.write_bytes(b"incomplete file")
            with self.assertRaisesRegex(ValueError, "invalid vision feature cache"):
                cache.load(key)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.model = EvoMindVLM(config(), vision_encoder=FakeEncoder(), processor=FakeProcessor())

    def test_frozen_encoder_stays_eval_and_projector_trains(self):
        self.model.train()
        self.assertFalse(self.model.vision_encoder.training)
        self.assertTrue(self.model.vision_proj.training)
        self.assertTrue(all(not p.requires_grad for p in self.model.vision_encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.vision_proj.parameters()))

    def test_cached_uncached_logits_loss_and_gradients_exact(self):
        for views in (1, 5):
            with self.subTest(views=views), tempfile.TemporaryDirectory() as directory:
                inputs = token_batch(views)
                pixels = torch.rand(1, views, 3, 8, 8)
                self.model.train()
                self.model.zero_grad(set_to_none=True)
                live = self.model(input_ids=inputs, labels=inputs, pixel_values=pixels)
                live.loss.backward()
                live_grad = {name: p.grad.clone() for name, p in self.model.vision_proj.named_parameters()}
                encoded = self.model.encode_images(pixels)
                cache = VisionFeatureCache(directory, "fake-frozen-encoder", "single" if views == 1 else "multi", hidden_size=8)
                key = cache.key_for(b"synthetic")
                cache.save(key, encoded[0])
                features = cache.load(key).unsqueeze(0)
                calls = self.model.vision_encoder.calls
                self.model.zero_grad(set_to_none=True)
                cached = self.model(input_ids=inputs, labels=inputs, vision_features=features)
                cached.loss.backward()
                self.assertEqual(calls, self.model.vision_encoder.calls)
                torch.testing.assert_close(cached.logits, live.logits, rtol=0, atol=0)
                torch.testing.assert_close(cached.loss, live.loss, rtol=0, atol=0)
                for name, parameter in self.model.vision_proj.named_parameters():
                    self.assertTrue(bool(parameter.grad.abs().sum() > 0), name)
                    torch.testing.assert_close(parameter.grad, live_grad[name], rtol=0, atol=0)
                self.assertTrue(all(p.grad is None for p in self.model.vision_encoder.parameters()))

    def test_matches_upstream_single_and_five_view_forward_and_decode(self):
        upstream = MiniMindVLM(config(), vision_model_path="__nonexistent_offline_test_encoder__")
        upstream.vision_encoder = FakeEncoder()
        upstream.load_state_dict(self.model.state_dict(), strict=True)
        upstream.eval()
        self.model.eval()
        for views in (1, 5):
            inputs = token_batch(views)
            values = {"pixel_values": torch.rand(1, views, 3, 8, 8)}
            with torch.no_grad():
                official = upstream(inputs, labels=inputs, pixel_values=values, use_cache=True)
                extended = self.model(inputs, labels=inputs, pixel_values=values, use_cache=True)
                torch.testing.assert_close(official.logits, extended.logits, rtol=0, atol=0)
                torch.testing.assert_close(official.loss, extended.loss, rtol=0, atol=0)
                next_token = torch.tensor([[7]])
                a = upstream(next_token, past_key_values=official.past_key_values, pixel_values=values)
                b = self.model(next_token, past_key_values=extended.past_key_values, pixel_values=values)
                torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)

    def test_exact_placeholder_segments_and_feature_shape_required(self):
        features = torch.rand(1, 5, 64, 8)
        for inputs in [token_batch(1), torch.tensor([[12] * 320]), token_batch(5)[:, :-5]]:
            with self.assertRaisesRegex(ValueError, "separate image placeholder"):
                self.model(inputs, vision_features=features)
        with self.assertRaisesRegex(ValueError, "trailing shape"):
            self.model(token_batch(1), vision_features=torch.zeros(1, 1, 63, 8))
        with self.assertRaisesRegex(ValueError, "require pixel_values"):
            self.model(token_batch(1))
        with self.assertRaisesRegex(ValueError, "either pixel_values"):
            self.model(token_batch(1), pixel_values=torch.rand(1, 1, 3, 8, 8), vision_features=features)
        with self.assertRaisesRegex(ValueError, "detached"):
            self.model(token_batch(5), vision_features=features.requires_grad_())

    def test_encoder_training_forbids_cache_and_live_frozen_path(self):
        self.model.vision_encoder.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen vision encoder"):
            self.model(token_batch(1), vision_features=torch.rand(1, 1, 64, 8))
        with self.assertRaisesRegex(ValueError, "frozen vision encoder"):
            self.model.encode_images(torch.rand(1, 1, 3, 8, 8))

    def test_outer_amp_does_not_change_encoder_compute_precision(self):
        pixels = torch.rand(1, 5, 3, 8, 8)
        expected = self.model.encode_images(pixels)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = self.model.encode_images(pixels)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.model.vision_encoder.bfloat16()
        with self.assertRaisesRegex(ValueError, "keep the frozen vision encoder in float32"):
            self.model.encode_images(pixels)

    def test_cache_only_constructor_and_generation_multiple_sequences(self):
        model = EvoMindVLM(config(), load_vision_encoder=False)
        self.assertIsNone(model.vision_encoder)
        features = self.model.encode_images(torch.rand(1, 1, 3, 8, 8))
        model.load_state_dict({name: value for name, value in self.model.state_dict().items()
                               if not name.startswith("vision_encoder.")}, strict=True)
        model.eval()
        generated = model.generate(token_batch(1), vision_features=features, max_new_tokens=2,
                                   num_return_sequences=2, do_sample=False, top_k=0, eos_token_id=None)
        self.assertEqual(tuple(generated.shape), (2, token_batch(1).shape[1] + 2))
        self.assertTrue(torch.equal(generated[0], generated[1]))


if __name__ == "__main__":
    unittest.main()
