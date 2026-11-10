from __future__ import annotations

import re
from abc import ABC, abstractmethod

from audiogen.codecs.base import CodecOutput
from audiogen.tokens.vocab import TokenMetadata


LEGACY_PROMPT_FORMAT = "legacy"
NATIVE_PROMPT_FORMAT = "speech_v1"


def resolve_prompt_policy(prompt_format="legacy", chat_template_kwargs=None):
    """Return a validated, copied policy; old checkpoints remain explicitly legacy."""
    if prompt_format not in (LEGACY_PROMPT_FORMAT, NATIVE_PROMPT_FORMAT):
        raise ValueError(f"Unknown prompt_format: {prompt_format!r}")
    kwargs = dict(chat_template_kwargs or {})
    if {"tokenize", "add_generation_prompt", "continue_final_message"} & kwargs.keys():
        raise ValueError("chat_template_kwargs cannot override prompt construction controls")
    if prompt_format == NATIVE_PROMPT_FORMAT:
        if kwargs.get("enable_thinking", False) is not False:
            raise ValueError("speech_v1 requires enable_thinking=False")
        kwargs["enable_thinking"] = False
    return prompt_format, kwargs


def prompt_policy_from_config(config):
    policy = getattr(config, "audiogen", None) or {}
    if not isinstance(policy, dict):
        raise ValueError("model.config.audiogen must be a dict")
    return resolve_prompt_policy(
        policy.get("prompt_format", LEGACY_PROMPT_FORMAT),
        policy.get("chat_template_kwargs"),
    )


def format_user_prompt(prompt: str, prompt_format: str = "legacy") -> str:
    resolve_prompt_policy(prompt_format)
    if prompt_format == LEGACY_PROMPT_FORMAT:
        return f"Generate audio: {prompt}"
    # Already formatted corpus prompts are used verbatim, including their tags.
    if re.match(r"^Generate speech audio(?:\s+\[[^\]]*\])*\s*:", prompt):
        return prompt
    return f"Generate speech audio: {prompt}"


def build_audio_prefix(
    tokenizer, prompt: str, metadata: TokenMetadata,
    prompt_format: str = "legacy", chat_template_kwargs: dict | None = None,
) -> list[int]:
    """Pure shared train/inference prefix, through (and including) <audio>.

    This never fetches a fallback template or mutates the tokenizer. Legacy
    inference may restore its historical Qwen template before calling here.
    """
    prompt_format, kwargs = resolve_prompt_policy(prompt_format, chat_template_kwargs)
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"{prompt_format} requires a saved tokenizer chat template")
    messages = [{"role": "user", "content": format_user_prompt(prompt, prompt_format)}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )
    if prompt_format == LEGACY_PROMPT_FORMAT:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError("Chat prefix tokenized to empty")
        return ids + [metadata.audio_start_token_id]
    ids = tokenizer.encode(text + "<audio>", add_special_tokens=False)
    if (ids.count(metadata.audio_start_token_id) != 1
            or not ids or ids[-1] != metadata.audio_start_token_id
            or metadata.audio_end_token_id in ids):
        raise ValueError("Native prefix must have exactly one <audio> boundary at its end")
    return ids


def validate_audio_stream(token_ids, metadata: TokenMetadata, max_new_tokens: int | None = None) -> dict:
    """Strict ID-level diagnostic of a continuation AFTER the opening <audio>.

    Unlike the legacy text parser, never ignores foreign IDs, reorders codebooks,
    or silently accepts incomplete frames. The vocab interval's end includes
    specials, so the codec range is computed from codebook count and size.
    """
    ids = token_ids.detach().cpu().tolist() if hasattr(token_ids, "detach") else list(token_ids)
    errors = []
    if any(type(i) is not int for i in ids):
        raise ValueError("audio stream IDs must be integers")
    cb, size, start = metadata.num_codebooks, metadata.codebook_size, metadata.audio_token_start_id
    if cb <= 0 or size <= 0:
        raise ValueError("Invalid audio codebook metadata")
    ends = [i for i, token in enumerate(ids) if token == metadata.audio_end_token_id]
    eos_seen = bool(ends)
    if len(ends) != 1:
        errors.append("expected exactly one audio EOS (</audio>)")
    if ends and ends[0] != len(ids) - 1:
        errors.append("tokens after audio EOS")
    body = ids[:ends[0]] if ends else ids
    flat, frames = [], 0
    if metadata.codec == "moss":
        for index, token in enumerate(body):
            relative = token - start
            if not 0 <= relative < cb * size:
                errors.append(f"token {index}: ID {token} outside codec range")
            elif relative // size != index % cb:
                errors.append(f"token {index}: codebook order mismatch")
            else:
                flat.append(relative % size)
        frames = len(body) // cb
        if len(body) % cb:
            errors.append("incomplete audio frame")
    elif metadata.codec == "snac":
        cursor, lengths = 0, []
        opens, closes = metadata.scale_open_ids or {}, metadata.scale_close_ids or {}
        for scale in range(cb):
            if cursor >= len(body) or body[cursor] != opens.get(scale):
                errors.append(f"missing or out-of-order scale {scale} opening")
                break
            cursor += 1
            length = 0
            while cursor < len(body) and body[cursor] != closes.get(scale):
                relative = body[cursor] - start - scale * size
                if not 0 <= relative < size:
                    errors.append(f"token {cursor}: invalid scale {scale} codec ID")
                else:
                    flat.append(relative)
                length += 1
                cursor += 1
            if cursor >= len(body):
                errors.append(f"missing scale {scale} closing")
                break
            cursor += 1
            lengths.append(length)
        if cursor != len(body):
            errors.append("unexpected tokens outside scale blocks")
        frames = lengths[0] if lengths else 0
        if lengths != [frames * (2 ** scale) for scale in range(cb)]:
            errors.append("incomplete SNAC frame ratios")
    else:
        raise ValueError(f"Unknown codec: {metadata.codec!r}")
    if not frames:
        errors.append("no complete audio frames")
    cap = max_new_tokens is not None and len(ids) >= max_new_tokens
    return {
        "valid": not errors, "errors": errors, "num_frames": frames,
        "num_tokens": len(ids), "eos_seen": eos_seen, "hit_token_cap": cap,
        "termination": "audio_eos" if eos_seen else ("token_cap" if cap else "missing_audio_eos"),
        "flat_tokens": flat,
    }


class TokenFormatter(ABC):
    @abstractmethod
    def codes_to_string(self, output: CodecOutput) -> str:
        """Convert a CodecOutput to a token string (without <audio> wrapper)."""
        ...

    @abstractmethod
    def string_to_flat_tokens(self, text: str) -> tuple[list[int], int]:
        """
        Parse a token string back to (flat_token_list, num_frames).
        The input may or may not have the <audio> wrapper.
        """
        ...

    def wrap(self, inner: str) -> str:
        return f"<audio>{inner}</audio>"

    def format_assistant_content(self, output: CodecOutput) -> str:
        return self.wrap(self.codes_to_string(output))

    def format_chat_messages(
        self, prompt: str, output: CodecOutput | None = None,
        prompt_format: str = "legacy",
    ) -> list[dict]:
        """
        Build chat messages with an explicit legacy or native prompt contract.
        If output is None, returns only the user turn (for inference — model completes from <audio>).
        """
        messages = [{"role": "user", "content": format_user_prompt(prompt, prompt_format)}]
        if output is not None:
            messages.append(
                {"role": "assistant", "content": self.format_assistant_content(output)}
            )
        return messages


class SNACTokenFormatter(TokenFormatter):
    def codes_to_string(self, output) -> str:
        # output.codes: list of 3 tensors [(1, T), (1, 2T), (1, 4T)]
        parts = []
        for s, scale_tensor in enumerate(output.codes):
            tokens_str = "".join(f"[AUDIO_S{s}_{v}]" for v in scale_tensor[0].tolist())
            parts.append(f"<s{s}>{tokens_str}</s{s}>")
        return "".join(parts)

    def string_to_flat_tokens(self, text: str) -> tuple[list[int], int]:
        text = text.replace("<audio>", "").replace("</audio>", "")
        flat = []
        scale0_len = 0
        for s in range(3):
            pattern = re.compile(rf"<s{s}>(.*?)</s{s}>", re.DOTALL)
            match = pattern.search(text)
            if not match:
                raise ValueError(f"Could not find scale {s} tokens in generated text")
            scale_tokens = re.findall(rf"\[AUDIO_S{s}_(\d+)\]", match.group(1))
            ids = [int(t) for t in scale_tokens]
            flat.extend(ids)
            if s == 0:
                scale0_len = len(ids)
        return flat, scale0_len


class MOSSTokenFormatter(TokenFormatter):
    def __init__(self, num_codebooks: int):
        self.num_codebooks = num_codebooks

    def codes_to_string(self, output) -> str:
        # output.codes: (num_codebooks, num_frames) — interleave by timestep
        tokens = output.codes.T.reshape(-1).tolist()
        cb = self.num_codebooks
        return "".join(
            f"[A{t % cb}_{v}]" for t, v in enumerate(tokens)
        )

    def string_to_flat_tokens(self, text: str) -> tuple[list[int], int]:
        text = text.replace("<audio>", "").replace("</audio>", "")
        cb = self.num_codebooks
        pattern = re.compile(r"\[A(\d+)_(\d+)\]")
        matches = pattern.findall(text)
        flat = [int(v) for _, v in matches]
        # Truncate to nearest complete frame
        num_frames = len(flat) // cb if cb > 0 else 0
        flat = flat[:num_frames * cb]
        return flat, num_frames


def get_formatter(codec: str, num_codebooks: int = 8) -> TokenFormatter:
    if codec == "snac":
        return SNACTokenFormatter()
    elif codec == "moss":
        return MOSSTokenFormatter(num_codebooks=num_codebooks)
    else:
        raise ValueError(f"Unknown codec: {codec!r}")
