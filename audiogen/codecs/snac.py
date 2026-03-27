from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio

from audiogen.codecs.base import CodecBase, CodecOutput

_DEFAULT_MODEL = "hubertsiuzdak/snac_44khz"
_SAMPLE_RATE = 44100
_CODEBOOK_SIZE = 4096
_NUM_SCALES = 3  # SNAC uses 3 hierarchical scales (1x, 2x, 4x)


@dataclass
class SNACOutput(CodecOutput):
    # codes: list of 3 tensors [(1, T), (1, 2T), (1, 4T)]
    scale_lengths: list[int]

    def to_flat_tokens(self) -> list[int]:
        """Concatenate all scales: [scale0_tokens..., scale1_tokens..., scale2_tokens...]"""
        tokens = []
        for scale_tensor in self.codes:
            tokens.extend(scale_tensor[0].tolist())
        return tokens

    @classmethod
    def from_flat_tokens(cls, tokens: list[int], num_codebooks: int, num_frames: int) -> "SNACOutput":
        # num_frames = scale0 length; scale1 = 2x, scale2 = 4x
        s0 = num_frames
        s1 = s0 * 2
        s2 = s0 * 4
        assert len(tokens) == s0 + s1 + s2, (
            f"Expected {s0 + s1 + s2} tokens for SNAC, got {len(tokens)}"
        )
        codes = [
            torch.tensor(tokens[:s0]).unsqueeze(0),
            torch.tensor(tokens[s0 : s0 + s1]).unsqueeze(0),
            torch.tensor(tokens[s0 + s1 :]).unsqueeze(0),
        ]
        return cls(codes=codes, num_frames=s0, num_codebooks=3, scale_lengths=[s0, s1, s2])


class SNACCodec(CodecBase):
    def __init__(self, model, device: torch.device):
        self._model = model
        self._device = device

    @property
    def sample_rate(self) -> int:
        return _SAMPLE_RATE

    @property
    def codebook_size(self) -> int:
        return _CODEBOOK_SIZE

    @property
    def num_codebooks(self) -> int:
        return _NUM_SCALES

    @classmethod
    def load(cls, model_path: str | None = None, device: str = "cpu", **kwargs) -> "SNACCodec":
        from snac import SNAC
        path = model_path or _DEFAULT_MODEL
        model = SNAC.from_pretrained(path).eval().to(device)
        return cls(model, torch.device(device))

    def _prepare(self, waveform: torch.Tensor) -> torch.Tensor:
        """Ensure mono, correct sample rate, correct shape (1, 1, samples)."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # waveform is now (1, samples) — add batch dim
        return waveform.unsqueeze(0).to(self._device)  # (1, 1, samples)

    def encode(self, waveform: torch.Tensor) -> SNACOutput:
        audio = self._prepare(waveform)
        with torch.no_grad():
            codes = self._model.encode(audio)
        # codes: list of 3 tensors [(1, T), (1, 2T), (1, 4T)]
        scale_lengths = [c.shape[-1] for c in codes]
        num_frames = scale_lengths[0]
        return SNACOutput(
            codes=codes,
            num_frames=num_frames,
            num_codebooks=_NUM_SCALES,
            scale_lengths=scale_lengths,
        )

    def decode(self, output: SNACOutput) -> torch.Tensor:
        # Align scale lengths to strict 1:2:4 ratio before decoding
        s0 = output.scale_lengths[0]
        codes = [
            output.codes[0][..., :s0],
            output.codes[1][..., : s0 * 2],
            output.codes[2][..., : s0 * 4],
        ]
        with torch.no_grad():
            audio = self._model.decode(codes)
        return audio.squeeze(0).cpu()  # (1, samples)
