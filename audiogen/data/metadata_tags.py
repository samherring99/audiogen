from __future__ import annotations

"""Normalization of Common Voice 17 self-reported metadata into closed-set prompt tags.

Philosophy: omit-on-doubt. A missing/unmappable field means the tag is absent from
the prompt, never an 'unknown' token.
"""

_GENDER_MAP = {
    "male_masculine": "male",
    "female_feminine": "female",
}

_AGE_BUCKETS = {"teens", "twenties", "thirties", "forties", "fifties", "sixties"}
_AGE_FIXES = {"fourties": "forties"}
_AGE_SPARSE = {"seventies", "eighties", "nineties"}

# Match on the first comma-token of the raw accent string, lowercased.
# Note: CV strings like "India and South Asia (India, Pakistan, Sri Lanka)" get
# comma-split too, so the first token may be a truncated "india and south asia (india".
_ACCENT_PREFIXES = [
    ("united states", "us"),
    ("england", "england"),
    ("india and south asia", "india_south_asia"),
    ("canadian", "canada"),
    ("german", "germanic_l2"),
    ("australian", "australia"),
    ("southern african", "southern_africa"),
    ("scottish", "scotland"),
    ("northern irish", "ireland"),
    ("irish", "ireland"),
    ("new zealand", "nz"),
    ("filipino", "filipino"),
    ("hong kong", "hong_kong"),
    ("singaporean", "singapore"),
]


def normalize_gender(raw: str | None) -> str | None:
    if not raw:
        return None
    return _GENDER_MAP.get(raw.strip().lower())


def normalize_age(raw: str | None) -> str | None:
    if not raw:
        return None
    value = raw.strip().lower()
    value = _AGE_FIXES.get(value, value)
    if value in _AGE_SPARSE:
        return "seventies_plus"
    if value in _AGE_BUCKETS:
        return value
    return None


def normalize_accent(raw: str | None) -> str | None:
    if not raw:
        return None
    first = raw.split(",")[0].strip().lower()
    if not first:
        return None
    for prefix, tag in _ACCENT_PREFIXES:
        if first.startswith(prefix):
            return tag
    return None


_PROMPT_PREFIX = "Generate speech audio"


def build_tagged_prompt(
    prompt: str,
    gender: str | None = None,
    age: str | None = None,
    accent: str | None = None,
) -> str:
    """Insert bracket tags between the instruction prefix and the ': <text>' payload.

    Fixed key order (gender, age, accent). No tags -> prompt returned unchanged,
    which is byte-identical to the current untagged format.
    """
    tags = []
    if gender:
        tags.append(f"[gender={gender}]")
    if age:
        tags.append(f"[age={age}]")
    if accent:
        tags.append(f"[accent={accent}]")
    if not tags:
        return prompt
    if not prompt.startswith(_PROMPT_PREFIX):
        raise ValueError(f"Unexpected prompt format: {prompt[:50]!r}")
    rest = prompt[len(_PROMPT_PREFIX):]
    return f"{_PROMPT_PREFIX} {' '.join(tags)}{rest}"
