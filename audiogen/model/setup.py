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
    old_vocab_size = len(tokenizer)
    # Preserve chat template — add_special_tokens can clear it in some transformers versions
    chat_template = getattr(tokenizer, "chat_template", None)
    tokenizer, meta = expand_tokenizer(tokenizer, codec, num_codebooks, codebook_size)
    if chat_template:
        tokenizer.chat_template = chat_template
    resize_model_embeddings(model, tokenizer)

    # Small random init for new tokens — zero causes instability on the lm_head side
    with torch.no_grad():
        std = model.model.embed_tokens.weight[:old_vocab_size].std().item()
        model.model.embed_tokens.weight[old_vocab_size:].normal_(mean=0, std=std * 0.01)
        model.lm_head.weight[old_vocab_size:].normal_(mean=0, std=std * 0.01)

    if save_dir is not None:
        save_dir = Path(save_dir)
        tokenizer.save_pretrained(save_dir)
        save_token_metadata(meta, save_dir / "audio_token_metadata.json")

    return model, tokenizer, meta


def load_trained_model(model_path: str | Path, base_model: str | None = None, device: str = "cpu"):
    model_path = Path(model_path)
    meta = load_token_metadata(model_path / "audio_token_metadata.json")

    # Rebuild tokenizer from base model and re-expand — saved tokenizers after add_tokens
    # can produce broken fast tokenizer files that encode everything to empty.
    if base_model is None:
        # Try to infer base model from saved config
        import json
        cfg_path = model_path / "config.json"
        try:
            cfg = json.loads(cfg_path.read_text())
            base_model = cfg.get("_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
        except Exception:
            base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        # Strip checkpoint suffix if base model path points to a fine-tuned dir
        if "/" not in base_model and not Path(base_model).exists():
            base_model = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Rebuilding tokenizer from base: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer, _ = expand_tokenizer(
        tokenizer, meta.codec, meta.num_codebooks, meta.codebook_size
    )

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model = model.to(device).eval()
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
