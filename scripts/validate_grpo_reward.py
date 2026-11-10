#!/usr/bin/env python3
"""Validate the GRPO whisper reward ranks known-quality audio correctly before training.

Uses real generated wavs with known human judgments + synthetic silence.
PASS requires: good samples > bad sample > silence, and digit-prompt scoring
works (9:45 canonicalization).
"""
import sys

import torch
import soundfile as sf

from audiogen.training.grpo import (
    _load_whisper,
    compute_whisper_reward,
    silence_penalty,
    tail_noise_penalty,
)

CASES = [
    # (wav path or "SILENCE"/"NOISE", prompt, human label)
    ("experiments/7b_sft_2026-07-31/audio/numbers_step1114/sample_0004.wav",
     "Generate speech audio: She finished 3rd in the race.", "good (user-approved)"),
    ("experiments/7b_sft_2026-07-31/audio/numbers_step1114/sample_0008.wav",
     "Generate speech audio: The train leaves from platform 7.", "good (user-approved)"),
    ("experiments/7b_sft_2026-07-31/audio/numbers_step1114/sample_0002.wav",
     "Generate speech audio: The meeting begins at 9:45.", "good at 1114 (digit-canon test)"),
    ("experiments/7b_sft_2026-07-31/audio/numbers_step400/sample_0000.wav",
     "Generate speech audio: I have 9 red apples.", "BAD (user: not good, low rms)"),
    ("SILENCE",
     "Generate speech audio: Hello there, how are you today?", "synthetic silence"),
    ("NOISE",
     "Generate speech audio: Hello there, how are you today?", "synthetic noise"),
    ("MISMATCH:experiments/7b_sft_2026-07-31/audio/numbers_step1114/sample_0008.wav",
     "Generate speech audio: The weather in Paris is lovely this evening.", "wrong-text control"),
]


def main() -> None:
    whisper = _load_whisper("openai/whisper-base")
    waveforms, prompts, labels = [], [], []
    for src, prompt, label in CASES:
        if src == "SILENCE":
            wav = torch.zeros(1, 24000 * 4)
        elif src == "NOISE":
            wav = torch.randn(1, 24000 * 4) * 0.1
        else:
            path = src.split(":", 1)[1] if src.startswith("MISMATCH:") else src
            data, sr = sf.read(path)
            wav = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        waveforms.append(wav)
        prompts.append(prompt)
        labels.append(label)

    rewards = compute_whisper_reward(waveforms, prompts, 24000, whisper)

    print(f"{'reward':>8}  {'sil_pen':>8}  {'tail_pen':>8}  label")
    for wav, r, label in zip(waveforms, rewards, labels):
        print(f"{r:8.3f}  {silence_penalty(wav):8.3f}  {tail_noise_penalty(wav):8.3f}  {label}")

    good = [r for r, l in zip(rewards, labels) if l.startswith("good")]
    bad = rewards[3]
    silence = rewards[4]
    noise = rewards[5]
    mismatch = rewards[6]

    checks = {
        "all good samples > bad sample": min(good) > bad,
        "bad sample > pure silence": bad > silence,
        "all good > silence": min(good) > silence,
        "all good > noise": min(good) > noise,
        "all good > wrong-text control": min(good) > mismatch,
        "digit 9:45 sample scores positive": rewards[2] > 0,
        "reward variance nontrivial": (max(rewards) - min(rewards)) > 0.5,
    }
    print()
    failed = False
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
        failed |= not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
