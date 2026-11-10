from __future__ import annotations

import json
from pathlib import Path

import torch
import soundfile as sf

from audiogen.codecs.base import CodecBase, CodecOutput
from audiogen.tokens.formatting import (
    TokenFormatter, build_audio_prefix, prompt_policy_from_config, validate_audio_stream,
)
from audiogen.tokens.vocab import TokenMetadata


def generate_token_ids(
    model,
    tokenizer,
    prompt: str,
    metadata: TokenMetadata,
    max_new_tokens: int = 1200,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: str = "cpu",
) -> torch.Tensor:
    config = getattr(model, "config", None)
    prompt_format, template_kwargs = prompt_policy_from_config(config)
    policy = getattr(config, "audiogen", None)
    model_type = getattr(config, "model_type", None)
    if (isinstance(model_type, str) and "nemotron" in model_type.lower() and
            (not policy or "prompt_format" not in policy)):
        raise ValueError("Nemotron requires an explicit audiogen prompt policy")
    source = (policy.get("base_model") if policy is not None
              else getattr(config, "_name_or_path", None))
    known_legacy_qwen = (
        prompt_format == "legacy" and model_type in ("qwen2", "qwen3") and
        isinstance(source, str) and source.startswith("Qwen/Qwen")
    )
    # Legacy formatting alone is not evidence of Qwen identity. Both its model
    # family and recorded source must match before historical template recovery.
    if not getattr(tokenizer, "chat_template", None) and known_legacy_qwen:
        from transformers import AutoTokenizer as _AT
        tokenizer.chat_template = _AT.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").chat_template
        print("Warning: restored missing legacy Qwen chat template")
    all_ids = build_audio_prefix(tokenizer, prompt, metadata, prompt_format, template_kwargs)
    input_ids = torch.tensor([all_ids], dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_ids)
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    input_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=metadata.audio_end_token_id,
        )

    return outputs[0][input_len:]  # only new tokens


def decode_token_ids(
    token_ids: torch.Tensor,
    tokenizer,
    formatter: TokenFormatter,
    metadata: TokenMetadata,
    strict: bool = False,
) -> CodecOutput | None:
    if strict:
        stream = validate_audio_stream(token_ids, metadata)
        if not stream["valid"]:
            raise ValueError(f"Invalid audio stream: {stream['errors']}")
        flat_tokens, num_frames = stream["flat_tokens"], stream["num_frames"]
    else:
        # Compatibility path: old MOSS parsing ignores codebook order and
        # truncates partial frames. Never use this to certify native output.
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        try:
            flat_tokens, num_frames = formatter.string_to_flat_tokens(text)
        except Exception as e:
            print(f"Warning: token parsing failed: {e}")
            return None
    if num_frames == 0:
        print("Warning: no audio tokens found in generated output — model may need more training")
        return None

    if metadata.codec == "snac":
        from audiogen.codecs.snac import SNACOutput
        return SNACOutput.from_flat_tokens(flat_tokens, metadata.num_codebooks, num_frames)
    else:
        from audiogen.codecs.moss import MOSSOutput
        return MOSSOutput.from_flat_tokens(flat_tokens, metadata.num_codebooks, num_frames)


def synthesize_audio(codec_output: CodecOutput, codec: CodecBase) -> torch.Tensor:
    return codec.decode(codec_output)  # (1, samples)


def generate_audio(
    model,
    tokenizer,
    codec: CodecBase,
    metadata: TokenMetadata,
    prompt: str,
    output_path: str | Path | None = None,
    max_new_tokens: int = 1200,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: str = "cpu",
    raw_token_path: str | Path | None = None,
    diagnostics: dict | None = None,
) -> torch.Tensor | None:
    from audiogen.tokens.formatting import get_formatter

    formatter = get_formatter(metadata.codec, metadata.num_codebooks)

    token_ids = generate_token_ids(
        model, tokenizer, prompt, metadata,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )

    prompt_format, template_kwargs = prompt_policy_from_config(getattr(model, "config", None))
    stream = validate_audio_stream(token_ids, metadata, max_new_tokens=max_new_tokens)
    report = {
        "prompt": prompt, "prompt_format": prompt_format,
        "chat_template_kwargs": template_kwargs,
        "raw_token_ids": token_ids.detach().cpu().tolist(), "stream": stream,
        "max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p,
        "decode_policy": "strict" if prompt_format == "speech_v1" else "legacy",
    }
    if diagnostics is not None:
        diagnostics.update(report)
    # Persist before parsing/codec decode so unsuccessful generations remain auditable.
    if raw_token_path is not None:
        raw_token_path = Path(raw_token_path)
        raw_token_path.parent.mkdir(parents=True, exist_ok=True)
        raw_token_path.write_text(json.dumps(report, indent=2) + "\n")
    if prompt_format == "speech_v1" and not stream["valid"]:
        print(f"Warning: invalid native audio stream: {stream['errors']}")
        return None
    codec_output = decode_token_ids(
        token_ids, tokenizer, formatter, metadata, strict=prompt_format == "speech_v1"
    )
    if codec_output is None:
        return None

    waveform = synthesize_audio(codec_output, codec)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_np = waveform.numpy().T  # (samples, channels)
        sf.write(str(output_path), audio_np, codec.sample_rate)

    return waveform
