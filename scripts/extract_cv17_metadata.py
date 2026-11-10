#!/usr/bin/env python3
"""Extract per-clip metadata from CV17 source tars into a sidecar JSONL.

Streams .metadata.json members only (no audio decode). Output rows:
{source_path, client_id, gender, age, accent} with normalized tag values
(None when missing/unmappable). source_path matches the encoded dataset's
format: "<tar_name>/<member_id>".
"""
from __future__ import annotations

import argparse
import collections
import json
import tarfile
from pathlib import Path

from audiogen.data.metadata_tags import normalize_accent, normalize_age, normalize_gender


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar_dir", required=True)
    parser.add_argument("--output", default="cv17_metadata.jsonl")
    args = parser.parse_args()

    tar_files = sorted(Path(args.tar_dir).glob("*.tar"))
    print(f"Found {len(tar_files)} tars", flush=True)

    stats = collections.Counter()
    speakers = set()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as out:
        for tar_path in tar_files:
            tar_name = tar_path.name
            n_tar = 0
            with tarfile.open(tar_path) as tar:
                for member in tar:
                    if not member.name.endswith(".metadata.json"):
                        continue
                    base_id = Path(member.name).name[: -len(".metadata.json")]
                    try:
                        meta = json.loads(tar.extractfile(member).read().decode("utf-8"))
                    except Exception:
                        stats["parse_fail"] += 1
                        continue
                    gender = normalize_gender(meta.get("gender"))
                    age = normalize_age(meta.get("age"))
                    accent = normalize_accent(meta.get("accent"))
                    client_id = meta.get("client_id") or None
                    if client_id:
                        speakers.add(client_id)
                    out.write(json.dumps({
                        "source_path": f"{tar_name}/{base_id}",
                        "client_id": client_id,
                        "gender": gender,
                        "age": age,
                        "accent": accent,
                    }) + "\n")
                    n_tar += 1
                    stats["rows"] += 1
                    stats["has_gender"] += gender is not None
                    stats["has_age"] += age is not None
                    stats["has_accent"] += accent is not None
                    stats["has_any"] += any(v is not None for v in (gender, age, accent))
            print(f"{tar_name}: {n_tar} rows", flush=True)

    total = stats["rows"] or 1
    summary = {
        "rows": stats["rows"],
        "unique_speakers": len(speakers),
        "parse_fail": stats["parse_fail"],
        "coverage_pct": {
            "gender": round(100 * stats["has_gender"] / total, 1),
            "age": round(100 * stats["has_age"] / total, 1),
            "accent": round(100 * stats["has_accent"] / total, 1),
            "any_tag": round(100 * stats["has_any"] / total, 1),
        },
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
