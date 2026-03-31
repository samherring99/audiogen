from __future__ import annotations

import re
from abc import ABC, abstractmethod

from audiogen.codecs.base import CodecOutput


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
        self, prompt: str, output: CodecOutput | None = None
    ) -> list[dict]:
        """
        Build chat messages for Qwen chat template.
        If output is None, returns only the user turn (for inference — model completes from <audio>).
        """
        messages = [{"role": "user", "content": f"Generate audio: {prompt}"}]
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
