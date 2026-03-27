from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class CodecOutput:
    codes: torch.Tensor  # (num_codebooks, num_frames) for MOSS; list of scale tensors for SNAC
    num_frames: int
    num_codebooks: int

    def to_flat_tokens(self) -> list[int]:
        raise NotImplementedError

    @classmethod
    def from_flat_tokens(cls, tokens: list[int], num_codebooks: int, num_frames: int) -> "CodecOutput":
        raise NotImplementedError


class CodecBase(ABC):
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @property
    @abstractmethod
    def codebook_size(self) -> int:
        ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int:
        ...

    @abstractmethod
    def encode(self, waveform: torch.Tensor) -> CodecOutput:
        """Encode a mono waveform (1, samples) at self.sample_rate into a CodecOutput."""
        ...

    @abstractmethod
    def decode(self, output: CodecOutput) -> torch.Tensor:
        """Decode a CodecOutput back to a mono waveform (1, samples)."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, model_path: str | None = None, device: str = "cpu", **kwargs) -> "CodecBase":
        ...
