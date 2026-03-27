from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio

from audiogen.codecs.base import CodecBase, CodecOutput

_DEFAULT_MODEL = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
_SAMPLE_RATE = 24000
_CODEBOOK_SIZE = 1024


@dataclass
class MOSSOutput(CodecOutput):
    # codes: (num_codebooks, num_frames)

    def to_flat_tokens(self) -> list[int]:
        """Interleave by timestep: [CB0_T0, CB1_T0, ..., CB{N-1}_T0, CB0_T1, ...]"""
        return self.codes.T.reshape(-1).tolist()

    @classmethod
    def from_flat_tokens(cls, tokens: list[int], num_codebooks: int, num_frames: int) -> "MOSSOutput":
        assert len(tokens) == num_codebooks * num_frames, (
            f"Expected {num_codebooks * num_frames} tokens, got {len(tokens)}"
        )
        codes = torch.tensor(tokens).reshape(num_frames, num_codebooks).T  # (num_codebooks, num_frames)
        return cls(codes=codes, num_frames=num_frames, num_codebooks=num_codebooks)


class MOSSCodec(CodecBase):
    def __init__(self, model, device: torch.device, n_codebooks: int = 8):
        self._model = model
        self._device = device
        self._num_codebooks = n_codebooks

    @property
    def sample_rate(self) -> int:
        return _SAMPLE_RATE

    @property
    def codebook_size(self) -> int:
        return _CODEBOOK_SIZE

    @property
    def num_codebooks(self) -> int:
        return self._num_codebooks

    @classmethod
    def load(
        cls,
        model_path: str | None = None,
        device: str = "cpu",
        num_codebooks: int = 8,
        dtype: torch.dtype | None = None,
        **kwargs,
    ) -> "MOSSCodec":
        from transformers import AutoModel
        path = model_path or _DEFAULT_MODEL

        # Let HF move weights to the right device/dtype rather than calling .to() after,
        # which can trigger recompilation issues on mismatched CUDA architectures.
        if dtype is not None:
            load_dtype = dtype
        elif device == "cpu":
            load_dtype = torch.float32  # CPU doesn't support float16/bfloat16 ops for this model
        else:
            load_dtype = torch.bfloat16
        model = AutoModel.from_pretrained(
            path, trust_remote_code=True, torch_dtype=load_dtype
        ).eval().to(device)
        return cls(model, torch.device(device), n_codebooks=num_codebooks)

    def _prepare(self, waveform: torch.Tensor) -> torch.Tensor:
        """Ensure mono, shape (1, 1, samples), matched to model device/dtype."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        model_dtype = next(self._model.parameters()).dtype
        return waveform.unsqueeze(0).to(device=self._device, dtype=model_dtype)

    def encode(self, waveform: torch.Tensor) -> MOSSOutput:
        audio = self._prepare(waveform)
        with torch.no_grad():
            output = self._model.encode(audio, num_quantizers=self._num_codebooks)
        codes = output.audio_codes[0]  # (num_codebooks, num_frames)
        return MOSSOutput(codes=codes.cpu(), num_frames=codes.shape[1], num_codebooks=self._num_codebooks)

    def decode(self, output: MOSSOutput) -> torch.Tensor:
        # codes: (num_codebooks, num_frames) -> (1, num_codebooks, num_frames)
        codes = output.codes.unsqueeze(0).to(self._device)
        with torch.no_grad():
            audio = self._model.decode(codes)
        return audio.squeeze(0).cpu()  # (1, samples)
