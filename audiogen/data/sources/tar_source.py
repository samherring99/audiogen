from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path
from typing import Iterator

import os
os.environ.setdefault("TORCHAUDIO_USE_BACKEND_DISPATCHER", "1")

import torch
import torchaudio

from audiogen.codecs.base import CodecBase
from audiogen.data.schema import AudioRecord


def iter_tar_examples(tar_path: str | Path) -> Iterator[tuple[bytes, str, str]]:
    """
    Yield (audio_bytes, transcript, source_id) from a Common Voice tar file.
    Expects files named: <id>.audio.mp3, <id>.sentence.txt, <id>.metadata.json
    """
    try:
        with tarfile.open(str(tar_path), "r:*") as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}

            ids: dict[str, dict] = {}
            for name in members:
                fname = Path(name).name
                if fname.endswith(".audio.mp3"):
                    base = fname[: -len(".audio.mp3")]
                    ids.setdefault(base, {})["mp3"] = name
                elif fname.endswith(".sentence.txt"):
                    base = fname[: -len(".sentence.txt")]
                    ids.setdefault(base, {})["txt"] = name

            for base_id, files in ids.items():
                if "mp3" not in files or "txt" not in files:
                    continue
                try:
                    audio_bytes = tar.extractfile(members[files["mp3"]]).read()
                    transcript = tar.extractfile(members[files["txt"]]).read().decode("utf-8").strip()
                    yield audio_bytes, transcript, base_id
                except Exception:
                    continue
    except Exception as e:
        print(f"Warning: failed to open {tar_path}: {e}", file=sys.stderr)


def process_tar(
    tar_path: str | Path,
    codec: CodecBase,
    min_duration_s: float = 1.0,
    max_duration_s: float = 30.0,
) -> list[AudioRecord]:
    records = []
    tar_name = Path(tar_path).name
    stats = {"found": 0, "load_fail": 0, "too_short": 0, "too_long": 0, "encode_fail": 0, "ok": 0}

    for audio_bytes, transcript, source_id in iter_tar_examples(tar_path):
        stats["found"] += 1
        try:
            import numpy as np
            import librosa
            audio_np, orig_sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=False)
            if audio_np.ndim == 1:
                audio_np = audio_np[np.newaxis, :]  # (1, samples)
            waveform = torch.from_numpy(audio_np)
        except Exception as e:
            print(f"Warning: audio load failed for {source_id}: {e}", file=sys.stderr)
            stats["load_fail"] += 1
            continue

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if orig_sr != codec.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_sr, codec.sample_rate)

        duration_s = waveform.shape[1] / codec.sample_rate
        if duration_s < min_duration_s:
            stats["too_short"] += 1
            continue
        if duration_s > max_duration_s:
            stats["too_long"] += 1
            continue

        try:
            output = codec.encode(waveform)
        except Exception as e:
            print(f"Warning: encode failed for {source_id}: {e}", file=sys.stderr)
            stats["encode_fail"] += 1
            continue

        stats["ok"] += 1
        records.append(
            AudioRecord(
                prompt=f"Generate speech audio: {transcript}",
                tokens=output.to_flat_tokens(),
                codec="moss" if not hasattr(output, "scale_lengths") else "snac",
                num_codebooks=output.num_codebooks,
                num_frames=output.num_frames,
                scale_lengths=getattr(output, "scale_lengths", None),
                source_path=f"{tar_name}/{source_id}",
                duration=round(duration_s, 2),
            )
        )

    if stats["found"] > 0:
        print(
            f"{tar_name}: found={stats['found']} ok={stats['ok']} "
            f"too_short={stats['too_short']} too_long={stats['too_long']} "
            f"load_fail={stats['load_fail']} encode_fail={stats['encode_fail']}",
            file=sys.stderr,
        )

    return records
