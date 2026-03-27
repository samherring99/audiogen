from __future__ import annotations

from pathlib import Path

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from audiogen.tokens.vocab import (
    TokenMetadata,
    expand_tokenizer,
    load_token_metadata,
    resize_model_embeddings,
    save_token_metadata,
)


def load_base_model(model_path: str, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model = model.to(device)
    return model, tokenizer


def expand_for_audio(
    model,
    tokenizer,
    codec: str,
    num_codebooks: int,
    codebook_size: int,
    save_dir: str | Path | None = None,
) -> tuple[object, object, TokenMetadata]:
    # Expand the tokenizer vocab with audio tokens and resize model embeddings.
    tokenizer, meta = expand_tokenizer(tokenizer, codec, num_codebooks, codebook_size)
    resize_model_embeddings(model, tokenizer)

    if save_dir is not None:
        save_dir = Path(save_dir)
        tokenizer.save_pretrained(save_dir)
        save_token_metadata(meta, save_dir / "audio_token_metadata.json")

    return model, tokenizer, meta


def load_trained_model(model_path: str | Path, device: str = "cpu"):
    model_path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model = model.to(device).eval()
    meta = load_token_metadata(model_path / "audio_token_metadata.json")
    return model, tokenizer, meta


def load_lora_model(base_model_path: str, lora_path: str | Path, device: str = "cpu"):
    lora_path = Path(lora_path)
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
    base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()
    model = model.to(device).eval()

    meta = load_token_metadata(lora_path / "audio_token_metadata.json")
    return model, tokenizer, meta
