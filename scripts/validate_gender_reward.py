#!/usr/bin/env python3
"""Validate the composite gender GRPO reward on real wavs before training.

Checks that the reward ranks: correct-gender speech > wrong-gender speech > silence,
using existing generated samples (voice gender per the validated classifier itself is
unknown for generated audio, so we use REAL CV17 clips with known speaker gender and
synthetic prompts claiming matching vs mismatching tags).
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import librosa
import torch

from audiogen.training.grpo import (
    _load_gender_classifier,
    _load_whisper,
    compute_gender_reward,
)

TAR_DIR = Path("/home/shared/omnimodel/common_voice_17")
METADATA = Path("data/cv17_metadata.jsonl")
N_PER_CASE = 6


def collect_clips(target_gender: str, n: int) -> list[tuple[torch.Tensor, str]]:
    """Return (wav_24k, transcript_prompt) for real clips of known speaker gender."""
    wanted = {}
    with METADATA.open() as f:
        for line in f:
            m = json.loads(line)
            if m.get("gender") == target_gender and m["source_path"].startswith("common-voice-17-000-000000.tar"):
                wanted[m["source_path"].split("/")[1]] = m
                if len(wanted) >= n * 3:
                    break
    out = []
    with tarfile.open(TAR_DIR / "common-voice-17-000-000000.tar") as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        for base, meta in wanted.items():
            if len(out) >= n:
                break
            mp3 = next((k for k in members if k.endswith(f"{base}.audio.mp3")), None)
            txt = next((k for k in members if k.endswith(f"{base}.sentence.txt")), None)
            if not mp3 or not txt:
                continue
            try:
                audio_bytes = tar.extractfile(members[mp3]).read()
                sentence = tar.extractfile(members[txt]).read().decode().strip()
                wav, _ = librosa.load(io.BytesIO(audio_bytes), sr=24000, mono=True)
                if len(wav) < 24000:
                    continue
                out.append((torch.tensor(wav).unsqueeze(0), sentence))
            except Exception:
                continue
    return out


def main() -> None:
    female = collect_clips("female", N_PER_CASE)
    male = collect_clips("male", N_PER_CASE)
    print(f"clips: female={len(female)} male={len(male)}", flush=True)

    waveforms, prompts, labels = [], [], []
    # Case A: female voice, prompt says female (MATCH)
    for wav, sent in female:
        waveforms.append(wav); prompts.append(f"Generate speech audio [gender=female]: {sent}"); labels.append("match_female")
    # Case B: female voice, prompt says male (MISMATCH)
    for wav, sent in female:
        waveforms.append(wav); prompts.append(f"Generate speech audio [gender=male]: {sent}"); labels.append("mismatch")
    # Case C: male voice, prompt says male (MATCH)
    for wav, sent in male:
        waveforms.append(wav); prompts.append(f"Generate speech audio [gender=male]: {sent}"); labels.append("match_male")
    # Case D: silence with a tag
    waveforms.append(torch.zeros(1, 24000 * 4)); prompts.append("Generate speech audio [gender=female]: Hello there."); labels.append("silence")

    gender_clf = _load_gender_classifier()
    whisper = _load_whisper("openai/whisper-base")
    rewards = compute_gender_reward(waveforms, prompts, 24000, gender_clf, whisper_tuple=whisper)

    import statistics as st
    groups = {}
    for r, l in zip(rewards, labels):
        groups.setdefault(l, []).append(r)
    means = {k: round(st.mean(v), 3) for k, v in groups.items()}
    print(json.dumps(means, indent=2))

    checks = {
        "match_female > mismatch": means["match_female"] > means["mismatch"],
        "match_male > mismatch": means["match_male"] > means["mismatch"],
        "mismatch > silence": means["mismatch"] > means["silence"],
        "match/mismatch gap > 0.3": (means["match_female"] + means["match_male"]) / 2 - means["mismatch"] > 0.3,
    }
    failed = False
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
        failed |= not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
