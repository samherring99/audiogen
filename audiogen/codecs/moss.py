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
    def __init__(self, model, device: torch.device, n_codebooks: int = 8, dtype: torch.dtype = torch.bfloat16):
        self._model = model
        self._device = device
        self._num_codebooks = n_codebooks
        self._dtype = dtype

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
        else:
            load_dtype = torch.float32  # bfloat16 causes internal dtype mismatch in MOSS decode
        model = AutoModel.from_pretrained(
            path, trust_remote_code=True, torch_dtype=load_dtype
        ).to(dtype=load_dtype).eval().to(device)
        return cls(model, torch.device(device), n_codebooks=num_codebooks, dtype=load_dtype)

    def _prepare(self, waveform: torch.Tensor) -> torch.Tensor:
        """Ensure mono, shape (1, 1, samples), matched to model device/dtype."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.unsqueeze(0).to(device=self._device, dtype=self._dtype)

    def encode(self, waveform: torch.Tensor) -> MOSSOutput:
        audio = self._prepare(waveform)
        with torch.no_grad():
            output = self._model.encode(audio, num_quantizers=self._num_codebooks)
        # audio_codes shape: (num_codebooks, batch, num_frames) -> squeeze batch -> (num_codebooks, num_frames)
        codes = output.audio_codes[:, 0, :]
        return MOSSOutput(codes=codes.cpu(), num_frames=codes.shape[1], num_codebooks=codes.shape[0])

    def decode(self, output: MOSSOutput) -> torch.Tensor:
        # model expects (num_codebooks, batch, num_frames)
        codes = output.codes.long().unsqueeze(1).to(self._device)
        print(f"[debug decode] codes shape: {codes.shape}, dtype: {codes.dtype}")
        with torch.no_grad():
            audio = self._model.decode(codes)
        return audio.squeeze(0).cpu()  # (1, samples)
