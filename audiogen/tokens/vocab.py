from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TokenMetadata:
    codec: str                    # "snac" or "moss"
    num_codebooks: int
    codebook_size: int
    audio_token_start_id: int     # first audio token id in vocab
    audio_token_end_id: int       # last audio token id + 1
    audio_start_token_id: int     # id of <audio>
    audio_end_token_id: int       # id of </audio>
    # SNAC only
    tokens_per_scale: int | None = None
    scale_open_ids: dict[int, int] | None = None   # scale_idx -> token_id of <sN>
    scale_close_ids: dict[int, int] | None = None  # scale_idx -> token_id of </sN>

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TokenMetadata":
        # Convert scale id dicts from JSON (string keys) back to int keys
        if d.get("scale_open_ids"):
            d["scale_open_ids"] = {int(k): v for k, v in d["scale_open_ids"].items()}
        if d.get("scale_close_ids"):
            d["scale_close_ids"] = {int(k): v for k, v in d["scale_close_ids"].items()}
        return cls(**d)


def build_snac_vocab(num_scales: int = 3, tokens_per_scale: int = 4096) -> list[str]:
    tokens = []
    for s in range(num_scales):
        for v in range(tokens_per_scale):
            tokens.append(f"[AUDIO_S{s}_{v}]")
    return tokens


def build_snac_special_tokens(num_scales: int = 3) -> list[str]:
    specials = ["<audio>", "</audio>"]
    for s in range(num_scales):
        specials += [f"<s{s}>", f"</s{s}>"]
    return specials


def build_moss_vocab(num_codebooks: int = 8, codebook_size: int = 1024) -> list[str]:
    tokens = []
    for c in range(num_codebooks):
        for v in range(codebook_size):
            tokens.append(f"[A{c}_{v}]")
    return tokens


def build_moss_special_tokens() -> list[str]:
    return ["<audio>", "</audio>"]


def expand_tokenizer(tokenizer, codec: str, num_codebooks: int, codebook_size: int):
    # Add audio tokens and special tokens to the tokenizer.
    if codec == "snac":
        audio_tokens = build_snac_vocab(num_scales=num_codebooks, tokens_per_scale=codebook_size)
        special_tokens = build_snac_special_tokens(num_scales=num_codebooks)
    elif codec == "moss":
        audio_tokens = build_moss_vocab(num_codebooks=num_codebooks, codebook_size=codebook_size)
        special_tokens = build_moss_special_tokens()
    else:
        raise ValueError(f"Unknown codec: {codec!r}")

    start_id = len(tokenizer)
    tokenizer.add_tokens(audio_tokens)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    end_id = len(tokenizer)

    audio_start_token_id = tokenizer.convert_tokens_to_ids("<audio>")
    audio_end_token_id = tokenizer.convert_tokens_to_ids("</audio>")

    meta = TokenMetadata(
        codec=codec,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        audio_token_start_id=start_id,
        audio_token_end_id=end_id,
        audio_start_token_id=audio_start_token_id,
        audio_end_token_id=audio_end_token_id,
    )

    if codec == "snac":
        meta.tokens_per_scale = codebook_size
        meta.scale_open_ids = {
            s: tokenizer.convert_tokens_to_ids(f"<s{s}>") for s in range(num_codebooks)
        }
        meta.scale_close_ids = {
            s: tokenizer.convert_tokens_to_ids(f"</s{s}>") for s in range(num_codebooks)
        }

    return tokenizer, meta


def resize_model_embeddings(model, tokenizer) -> None:
    model.resize_token_embeddings(len(tokenizer))


def save_token_metadata(meta: TokenMetadata, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta.to_dict(), f, indent=2)


def load_token_metadata(path: Path | str) -> TokenMetadata:
    with open(path) as f:
        return TokenMetadata.from_dict(json.load(f))
