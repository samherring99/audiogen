from __future__ import annotations

"""Build the metadata-conditioned training mixture from existing encoded records.

Prompt rewriting only — audio tokens are never modified. Variants:
  full:    all available normalized tags
  partial: random non-empty subset of available tags
  none:    untagged (identical to current prompt format)
Speaker-disjoint eval split by client_id.
"""

import copy
import random

from audiogen.data.metadata_tags import build_tagged_prompt

VARIANT_WEIGHTS = [("full", 0.6), ("partial", 0.2), ("none", 0.2)]


def assign_prompt_variant(prompt: str, meta: dict, mode: str, rng: random.Random | None = None) -> str:
    gender = meta.get("gender")
    age = meta.get("age")
    accent = meta.get("accent")
    if mode == "none":
        return prompt
    if mode == "full":
        return build_tagged_prompt(prompt, gender=gender, age=age, accent=accent)
    if mode == "partial":
        rng = rng or random.Random(0)
        available = [(k, v) for k, v in (("gender", gender), ("age", age), ("accent", accent)) if v]
        if not available:
            return prompt
        k = rng.randint(1, len(available))
        chosen = dict(rng.sample(available, k))
        return build_tagged_prompt(
            prompt,
            gender=chosen.get("gender"),
            age=chosen.get("age"),
            accent=chosen.get("accent"),
        )
    raise ValueError(f"Unknown mode: {mode}")


def _pick_variant(rng: random.Random) -> str:
    x = rng.random()
    acc = 0.0
    for name, w in VARIANT_WEIGHTS:
        acc += w
        if x < acc:
            return name
    return VARIANT_WEIGHTS[-1][0]


def make_metadata_dataset(
    records: list[dict],
    metadata: dict[str, dict],
    target_size: int,
    eval_speaker_fraction: float = 0.02,
    per_speaker_cap: int = 50,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)

    # Only records with metadata AND at least one usable tag participate in tagging;
    # records without metadata can still appear as 'none' variants.
    joined = []
    for rec in records:
        meta = metadata.get(rec.get("source_path"))
        if meta is None:
            continue
        joined.append((rec, meta))

    # Speaker-disjoint eval split
    speakers = sorted({m["client_id"] for _, m in joined if m.get("client_id")})
    rng.shuffle(speakers)
    n_eval = max(1, int(len(speakers) * eval_speaker_fraction))
    eval_speakers = set(speakers[:n_eval])

    train_pool = [(r, m) for r, m in joined if m.get("client_id") and m["client_id"] not in eval_speakers]
    eval_pool = [(r, m) for r, m in joined if m.get("client_id") in eval_speakers]

    rng.shuffle(train_pool)

    # Per-speaker cap + target size
    counts: dict[str, int] = {}
    train_records = []
    train_manifest = []
    for rec, meta in train_pool:
        cid = meta["client_id"]
        if counts.get(cid, 0) >= per_speaker_cap:
            continue
        variant = _pick_variant(rng)
        prompt = assign_prompt_variant(rec["prompt"], meta, variant, rng)
        out = copy.deepcopy(rec)
        out["prompt"] = prompt
        train_records.append(out)
        train_manifest.append({
            "source_path": rec["source_path"],
            "client_id": cid,
            "variant": variant,
            "gender": meta.get("gender"),
            "age": meta.get("age"),
            "accent": meta.get("accent"),
            "prompt": prompt,
        })
        counts[cid] = counts.get(cid, 0) + 1
        if len(train_records) >= target_size:
            break

    eval_manifest = []
    for rec, meta in eval_pool:
        eval_manifest.append({
            "source_path": rec["source_path"],
            "client_id": meta["client_id"],
            "gender": meta.get("gender"),
            "age": meta.get("age"),
            "accent": meta.get("accent"),
            "prompt_untagged": rec["prompt"],
            "prompt_tagged": assign_prompt_variant(rec["prompt"], meta, "full"),
        })

    return {
        "train_records": train_records,
        "train_manifest": train_manifest,
        "eval_manifest": eval_manifest,
        "eval_speakers": sorted(eval_speakers),
    }
