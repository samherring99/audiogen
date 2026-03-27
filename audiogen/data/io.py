from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from audiogen.data.schema import AudioRecord


def load_records(path: str | Path) -> list[AudioRecord]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return _load_jsonl(path)
    elif path.suffix == ".json":
        return _load_json(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}. Use .json or .jsonl")


def _load_jsonl(path: Path) -> list[AudioRecord]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records.append(_autodetect_record(d))
    return records


def _load_json(path: Path) -> list[AudioRecord]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return [_autodetect_record(d) for d in data]
    raise ValueError("Expected a JSON array at top level")


def _autodetect_record(d: dict) -> AudioRecord:
    if "audio_tokens" in d:
        return AudioRecord.from_snac_dict(d)
    elif "tokens" in d:
        return AudioRecord.from_moss_dict(d)
    else:
        raise ValueError(f"Cannot determine codec from record keys: {list(d.keys())}")


def save_records_jsonl(records: list[AudioRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r.to_moss_dict()) + "\n")


def save_records_json(records: list[AudioRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([r.to_snac_dict() for r in records], f, indent=2)


def split_records(
    records: list[AudioRecord], val_split: float = 0.05, seed: int = 42
) -> tuple[list[AudioRecord], list[AudioRecord]]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split))
    return shuffled[n_val:], shuffled[:n_val]


def merge_datasets(input_dirs: list[str | Path], output_path: str | Path) -> list[AudioRecord]:
    all_records: list[AudioRecord] = []
    for d in input_dirs:
        d = Path(d)
        candidates = list(d.glob("*.jsonl")) + list(d.glob("*.json")) if d.is_dir() else [d]
        for p in candidates:
            all_records.extend(load_records(p))
    save_records_jsonl(all_records, output_path)
    return all_records


def load_prompts(path: str | Path) -> list[str]:
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        if data and isinstance(data[0], str):
            return data
        # list of dicts with "prompt" key
        return [d["prompt"] for d in data if "prompt" in d]
    raise ValueError("Expected a JSON array of strings or dicts with 'prompt' key")


def save_prompts(prompts: list[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(prompts, f, indent=2)
