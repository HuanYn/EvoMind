"""A transparent character-level tokenizer for the MiniMind learning project."""

from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    """Character tokenizer with a stable vocabulary and four special tokens."""

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"
    SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)

    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {token_id: token for token, token_id in self.token_to_id.items()}

        if set(self.id_to_token) != set(range(len(self.token_to_id))):
            raise ValueError("token IDs must be unique and contiguous from 0")
        for token in self.SPECIAL_TOKENS:
            if token not in self.token_to_id:
                raise ValueError(f"missing required special token: {token}")

    @classmethod
    def build(cls, text: str) -> "CharTokenizer":
        characters = sorted(set(text) - set(cls.SPECIAL_TOKENS))
        tokens = list(cls.SPECIAL_TOKENS) + characters
        return cls({token: index for index, token in enumerate(tokens)})

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK_TOKEN]

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.token_to_id.get(character, self.unk_id) for character in text]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        tokens: list[str] = []
        for token_id in ids:
            token = self.id_to_token.get(int(token_id), self.UNK_TOKEN)
            if skip_special_tokens and token in self.SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return "".join(tokens)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"token_to_id": self.token_to_id}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["token_to_id"])
