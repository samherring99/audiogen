from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AudioRecord:
    prompt: str
    tokens: list[int]
    codec: Literal["snac", "moss"]
    num_codebooks: int
    num_frames: int
    scale_lengths: list[int] | None = None   # SNAC only
    source_path: str | None = None
    audio_path: str | None = None
    duration: float | None = None

    @classmethod
    def from_snac_dict(cls, d: dict) -> "AudioRecord":
        scale_lengths = d.get("scale_lengths")
        num_frames = scale_lengths[0] if scale_lengths else 0
        return cls(
            prompt=d["prompt"],
            tokens=d["audio_tokens"],
            codec="snac",
            num_codebooks=3,
            num_frames=num_frames,
            scale_lengths=scale_lengths,
            audio_path=d.get("audio_path"),
            source_path=d.get("source_path"),
        )

    @classmethod
    def from_moss_dict(cls, d: dict) -> "AudioRecord":
        return cls(
            prompt=d["prompt"],
            tokens=d["tokens"],
            codec="moss",
            num_codebooks=d["num_codebooks"],
            num_frames=d["num_frames"],
            source_path=d.get("source_path"),
        )

    def to_snac_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "audio_tokens": self.tokens,
            "scale_lengths": self.scale_lengths,
            "audio_path": self.audio_path,
            "source_path": self.source_path,
        }

    def to_moss_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "tokens": self.tokens,
            "num_codebooks": self.num_codebooks,
            "num_frames": self.num_frames,
            "source_path": self.source_path,
        }
