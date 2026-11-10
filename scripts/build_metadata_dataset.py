#!/usr/bin/env python3
"""Build the metadata-conditioned training dataset from the 1M anchor + sidecar.

Output dir contains:
  train.jsonl        — training records (prompt-rewritten, tokens untouched)
  train_manifest.jsonl — provenance: variant, tags, client_id per record
  eval_manifest.json — held-out speaker eval prompts (tagged + untagged forms)
  stats.json         — audit numbers
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from audiogen.data.metadata_dataset import make_metadata_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="data/cv17_en_moss_1m.jsonl")
    parser.add_argument("--metadata", default="data/cv17_metadata.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_size", type=int, default=175000)
    parser.add_argument("--eval_speaker_fraction", type=float, default=0.02)
    parser.add_argument("--per_speaker_cap", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading metadata sidecar...", flush=True)
    metadata = {}
    with open(args.metadata) as f:
        for line in f:
            m = json.loads(line)
            metadata[m["source_path"]] = m

    print("Loading encoded records...", flush=True)
    records = []
    with open(args.records) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"{len(records)} records, {len(metadata)} metadata rows. Building...", flush=True)
    result = make_metadata_dataset(
        records, metadata,
        target_size=args.target_size,
        eval_speaker_fraction=args.eval_speaker_fraction,
        per_speaker_cap=args.per_speaker_cap,
        seed=args.seed,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "train.jsonl").open("w") as f:
        for r in result["train_records"]:
            f.write(json.dumps(r) + "\n")
    with (out / "train_manifest.jsonl").open("w") as f:
        for r in result["train_manifest"]:
            f.write(json.dumps(r) + "\n")
    (out / "eval_manifest.json").write_text(json.dumps(result["eval_manifest"][:2000], indent=2) + "\n")

    manifest = result["train_manifest"]
    variants = collections.Counter(r["variant"] for r in manifest)
    genders = collections.Counter(r["gender"] for r in manifest if r["variant"] != "none")
    ages = collections.Counter(r["age"] for r in manifest if r["variant"] != "none")
    accents = collections.Counter(r["accent"] for r in manifest if r["variant"] != "none")
    speakers = collections.Counter(r["client_id"] for r in manifest)
    train_speaker_set = set(speakers)
    eval_speaker_set = {r["client_id"] for r in result["eval_manifest"]}

    stats = {
        "train_records": len(result["train_records"]),
        "train_speakers": len(train_speaker_set),
        "eval_speakers_held_out": len(result["eval_speakers"]),
        "eval_manifest_rows_written": min(len(result["eval_manifest"]), 2000),
        "speaker_overlap_train_eval": len(train_speaker_set & eval_speaker_set),
        "max_clips_per_speaker": max(speakers.values()),
        "variant_mix": dict(variants),
        "gender_dist_tagged": dict(genders.most_common()),
        "age_dist_tagged": dict(ages.most_common()),
        "accent_dist_tagged": dict(accents.most_common()),
        "seed": args.seed,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
