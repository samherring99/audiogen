#!/usr/bin/env python3
"""Validate a pretrained gender classifier against CV17 self-reported ground truth.

Samples real clips (known gender from sidecar metadata) from source tars, runs the
classifier, reports accuracy + confusion. Gate: >=90% balanced accuracy to admit
the classifier into any GRPO reward.
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

MODEL = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar_dir", default="/home/shared/omnimodel/common_voice_17")
    parser.add_argument("--metadata", default="data/cv17_metadata.jsonl")
    parser.add_argument("--per_class", type=int, default=250)
    parser.add_argument("--num_tars", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="experiments/7b_sft_2026-07-31/gender_classifier_validation.json")
    args = parser.parse_args()

    # Collect labeled source_paths from the first N tars
    wanted_tars = sorted(Path(args.tar_dir).glob("*.tar"))[: args.num_tars]
    tar_names = {t.name for t in wanted_tars}
    labels: dict[str, str] = {}
    per_class = Counter()
    with open(args.metadata) as f:
        for line in f:
            m = json.loads(line)
            g = m.get("gender")
            if g not in ("male", "female"):
                continue
            tar_name = m["source_path"].split("/")[0]
            if tar_name not in tar_names:
                continue
            if per_class[g] >= args.per_class * 2:  # oversample pool, subsample later
                continue
            labels[m["source_path"]] = g
            per_class[g] += 1
    print(f"Candidate labeled clips: {dict(per_class)}", flush=True)

    from transformers import pipeline
    clf = pipeline("audio-classification", model=MODEL, device=0 if args.device == "cuda" else -1)

    import librosa
    results = []
    done = Counter()
    for tar_path in wanted_tars:
        with tarfile.open(tar_path) as tar:
            members = {m.name: m for m in tar.getmembers() if m.name.endswith(".audio.mp3")}
            for name, member in members.items():
                base = Path(name).name[: -len(".audio.mp3")]
                sp = f"{tar_path.name}/{base}"
                g = labels.get(sp)
                if g is None or done[g] >= args.per_class:
                    continue
                try:
                    audio_bytes = tar.extractfile(member).read()
                    wav, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
                    if len(wav) < 16000:  # skip <1s clips
                        continue
                    pred = clf({"array": wav, "sampling_rate": 16000}, top_k=2)
                    top = max(pred, key=lambda x: x["score"])
                    pred_label = top["label"].lower()
                    results.append({"source_path": sp, "true": g, "pred": pred_label, "score": top["score"]})
                    done[g] += 1
                except Exception as e:
                    print(f"skip {sp}: {e}", flush=True)
        if all(done[c] >= args.per_class for c in ("male", "female")):
            break

    # Score. Model labels are 'male'/'female' already; be defensive about casing/variants.
    def norm(label: str) -> str:
        label = label.lower()
        if "female" in label or label in ("f", "woman"):
            return "female"
        if "male" in label or label in ("m", "man"):
            return "male"
        return label

    confusion = Counter()
    for r in results:
        confusion[(r["true"], norm(r["pred"]))] += 1
    per_class_acc = {}
    for cls in ("male", "female"):
        total = sum(v for (t, _), v in confusion.items() if t == cls)
        correct = confusion.get((cls, cls), 0)
        per_class_acc[cls] = round(correct / total, 4) if total else None
    balanced = round(sum(v for v in per_class_acc.values() if v is not None) / 2, 4)

    summary = {
        "model": MODEL,
        "n_scored": len(results),
        "per_class_accuracy": per_class_acc,
        "balanced_accuracy": balanced,
        "confusion": {f"{t}->{p}": v for (t, p), v in sorted(confusion.items())},
        "mean_confidence": round(float(np.mean([r["score"] for r in results])), 4) if results else None,
        "verdict": "PASS (>=0.90)" if balanced and balanced >= 0.90 else "FAIL (<0.90)",
    }
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
