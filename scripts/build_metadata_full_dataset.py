#!/usr/bin/env python3
"""Build the FULL-SCALE tagged dataset for a fresh phase-2 style conditioned train.

Unlike build_metadata_dataset.py (175k focused mixture for post-hoc adaptation),
this keeps ALL 1M anchor records at their original scale and only rewrites prompts:
  - records with >=1 usable tag: 70% full tags / 15% partial / 15% untagged
  - records with no usable metadata: passed through unchanged
Tokens/audio are never modified. Output is a drop-in replacement for
data/cv17_en_moss_1m.jsonl in the proven phase-2 recipe.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from audiogen.data.metadata_dataset import assign_prompt_variant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="data/cv17_en_moss_1m.jsonl")
    parser.add_argument("--metadata", default="data/cv17_metadata.jsonl")
    parser.add_argument("--output", default="data/cv17_en_moss_1m_tagged.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("Loading metadata sidecar...", flush=True)
    metadata = {}
    with open(args.metadata) as f:
        for line in f:
            m = json.loads(line)
            metadata[m["source_path"]] = m

    stats = collections.Counter()
    tag_counts = collections.Counter()
    out_path = Path(args.output)
    with open(args.records) as fin, out_path.open("w") as fout:
        for line in fin:
            record = json.loads(line)
            stats["total"] += 1
            meta = metadata.get(record.get("source_path"))
            if meta is not None and any(meta.get(k) for k in ("gender", "age", "accent")):
                roll = rng.random()
                if roll < 0.70:
                    mode = "full"
                elif roll < 0.85:
                    mode = "partial"
                else:
                    mode = "none"
                new_prompt = assign_prompt_variant(record["prompt"], meta, mode, rng)
                if new_prompt != record["prompt"]:
                    stats[f"variant_{mode}"] += 1
                    for k in ("gender", "age", "accent"):
                        if meta.get(k) and f"[{k}=" in new_prompt:
                            tag_counts[f"{k}={meta[k]}"] += 1
                else:
                    stats["variant_none"] += 1
                record["prompt"] = new_prompt
            else:
                stats["no_metadata_passthrough"] += 1
            fout.write(json.dumps(record) + "\n")
            if stats["total"] % 200000 == 0:
                print(f"...{stats['total']}", flush=True)

    summary = {
        "output": str(out_path),
        "total_records": stats["total"],
        "tagged_full": stats["variant_full"],
        "tagged_partial": stats["variant_partial"],
        "untagged_by_dropout": stats["variant_none"],
        "untagged_no_metadata": stats["no_metadata_passthrough"],
        "top_tags": dict(tag_counts.most_common(20)),
        "seed": args.seed,
    }
    Path(str(out_path) + ".stats.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
