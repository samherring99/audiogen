#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from audiogen.data.number_variants import build_number_variant, classify_safe_variant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pair_count", type=int, default=25000)
    parser.add_argument("--replay_count", type=int, default=25000)
    parser.add_argument("--eval_count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    candidates = []
    replay_reservoir = []
    total = 0
    category_counts = collections.Counter()

    with open(args.input) as handle:
        for line in handle:
            record = json.loads(line)
            total += 1
            variant = build_number_variant(record["prompt"])
            if variant:
                category = classify_safe_variant(record["prompt"])
                candidates.append((record, variant, category))
                category_counts[category] += 1
            # Reservoir sample enough replay candidates before excluding pair/eval sources.
            reservoir_size = args.replay_count * 3
            if len(replay_reservoir) < reservoir_size:
                replay_reservoir.append(record)
            else:
                j = rng.randrange(total)
                if j < reservoir_size:
                    replay_reservoir[j] = record

    rng.shuffle(candidates)
    if len(candidates) < args.pair_count + args.eval_count:
        raise RuntimeError(
            f"Only {len(candidates)} safe candidates; need {args.pair_count + args.eval_count}"
        )

    eval_candidates = candidates[: args.eval_count]
    train_candidates = candidates[args.eval_count : args.eval_count + args.pair_count]
    heldout_sources = {r.get("source_path") for r, _, _ in eval_candidates}
    train_sources = {r.get("source_path") for r, _, _ in train_candidates}

    train_records = []
    provenance_path = output_dir / "number_pairs.jsonl"
    with provenance_path.open("w") as provenance:
        for record, variant, category in train_candidates:
            word_record = dict(record)
            digit_record = dict(record)
            digit_record["prompt"] = variant
            train_records.extend((word_record, digit_record))
            provenance.write(json.dumps({
                "source_path": record.get("source_path"),
                "category": category,
                "word_prompt": record["prompt"],
                "digit_prompt": variant,
            }) + "\n")

    replay = [
        r for r in replay_reservoir
        if r.get("source_path") not in heldout_sources | train_sources
    ][: args.replay_count]
    if len(replay) < args.replay_count:
        raise RuntimeError(f"Only {len(replay)} replay records after source exclusion")
    train_records.extend(replay)
    rng.shuffle(train_records)

    train_path = output_dir / "train.jsonl"
    with train_path.open("w") as handle:
        for record in train_records:
            handle.write(json.dumps(record) + "\n")

    eval_manifest = []
    for record, variant, category in eval_candidates:
        eval_manifest.append({
            "source_path": record.get("source_path"),
            "category": category,
            "word_prompt": record["prompt"],
            "digit_prompt": variant,
        })
    (output_dir / "numeric_eval.json").write_text(json.dumps(eval_manifest, indent=2) + "\n")

    stats = {
        "input": args.input,
        "input_records": total,
        "safe_candidates": len(candidates),
        "candidate_categories": dict(category_counts),
        "pair_count": len(train_candidates),
        "paired_train_records": len(train_candidates) * 2,
        "replay_records": len(replay),
        "total_train_records": len(train_records),
        "numeric_eval_pairs": len(eval_manifest),
        "seed": args.seed,
        "source_disjoint": not (
            {r.get("source_path") for r in train_records} & heldout_sources
        ),
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
