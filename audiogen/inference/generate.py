from __future__ import annotations

from pathlib import Path

import torch
import soundfile as sf
import numpy as np

from audiogen.codecs.base import CodecBase, CodecOutput
from audiogen.tokens.formatting import TokenFormatter
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
    from audiogen.tokens.formatting import get_formatter

    formatter = get_formatter(metadata.codec, metadata.num_codebooks)
    messages = formatter.format_chat_messages(prompt, output=None)

    # If the saved tokenizer lost its chat template (happens when add_special_tokens clears it),
    # fall back to loading it from the original Qwen instruct model.
    if tokenizer.chat_template is None:
        from transformers import AutoTokenizer as _AT
        _base = _AT.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        tokenizer.chat_template = _base.chat_template
        print("Warning: tokenizer.chat_template was None — loaded from Qwen/Qwen2.5-0.5B-Instruct")

    # Apply chat template and add the opening <audio> tag to prime generation
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text += "<audio>"

    print(f"[debug] prompt text: {repr(text[:200])}")
    # Use encode directly — some saved tokenizers fail with __call__ but encode works
    prefix_text = text[:-len("<audio>")]
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    print(f"[debug] prefix token count: {len(prefix_ids)}")
    assert len(prefix_ids) > 0, f"Prefix tokenized to empty. Check tokenizer. Text: {repr(prefix_text[:100])}"
    audio_start_id = metadata.audio_start_token_id
    all_ids = prefix_ids + [audio_start_id]
    input_ids = torch.tensor([all_ids], dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_ids)
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    print(f"[debug] input_ids shape: {input_ids.shape}")
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
) -> CodecOutput | None:
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    print(f"[debug] generated text ({len(token_ids)} tokens): {repr(text[:300])}")
    try:
        flat_tokens, num_frames = formatter.string_to_flat_tokens(text)
    except Exception as e:
        print(f"Warning: token parsing failed: {e}")
        return None
    print(f"[debug] parsed {len(flat_tokens)} flat tokens, {num_frames} frames")
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

    codec_output = decode_token_ids(token_ids, tokenizer, formatter, metadata)
    if codec_output is None:
        return None

    waveform = synthesize_audio(codec_output, codec)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_np = waveform.numpy().T  # (samples, channels)
        sf.write(str(output_path), audio_np, codec.sample_rate)

    return waveform
