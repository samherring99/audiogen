from __future__ import annotations

import copy
import random
import re
from typing import Iterable

_SMALL = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_ORDINALS = {
    "first": (1, "1st"),
    "second": (2, "2nd"),
    "third": (3, "3rd"),
    "fourth": (4, "4th"),
    "fifth": (5, "5th"),
    "sixth": (6, "6th"),
    "seventh": (7, "7th"),
    "eighth": (8, "8th"),
    "ninth": (9, "9th"),
    "tenth": (10, "10th"),
    "eleventh": (11, "11th"),
    "twelfth": (12, "12th"),
    "thirteenth": (13, "13th"),
    "fourteenth": (14, "14th"),
    "fifteenth": (15, "15th"),
    "sixteenth": (16, "16th"),
    "seventeenth": (17, "17th"),
    "eighteenth": (18, "18th"),
    "nineteenth": (19, "19th"),
    "twentieth": (20, "20th"),
    "thirtieth": (30, "30th"),
    "fortieth": (40, "40th"),
    "fiftieth": (50, "50th"),
    "sixtieth": (60, "60th"),
    "seventieth": (70, "70th"),
    "eightieth": (80, "80th"),
    "ninetieth": (90, "90th"),
}
_CARDINAL_WORDS = set(_SMALL) | set(_TENS)
_WORD_BOUNDARY = r"(?<![A-Za-z]){}(?![A-Za-z])"


def _preserve_initial_case(original: str, replacement: str) -> str:
    return replacement if not original[:1].isupper() else replacement


def _clock_replacement(match: re.Match[str]) -> str:
    hour_word = match.group("hour").lower()
    minute_tens = match.group("minute_tens").lower()
    minute_small = match.group("minute_small").lower()
    hour = _SMALL.get(hour_word, _TENS.get(hour_word))
    minute = _TENS[minute_tens] + _SMALL[minute_small]
    return f"{match.group('prefix')}{hour}:{minute:02d}"


def classify_safe_variant(text: str) -> str | None:
    if _time_pattern().search(text):
        return "time"
    if _ordinal_pattern().search(text):
        return "ordinal"
    if _compound_pattern().search(text) or _single_pattern().search(text):
        return "cardinal"
    return None


def _time_pattern() -> re.Pattern[str]:
    hours = "|".join([w for w, value in _SMALL.items() if 1 <= value <= 12])
    tens = "|".join(_TENS)
    small_minutes = "|".join([w for w, value in _SMALL.items() if 1 <= value <= 9])
    return re.compile(
        rf"(?P<prefix>\bat\s+)(?P<hour>{hours})[ -]+(?P<minute_tens>{tens})[ -]+(?P<minute_small>{small_minutes})\b",
        re.IGNORECASE,
    )


def _ordinal_pattern() -> re.Pattern[str]:
    return re.compile(_WORD_BOUNDARY.format("(" + "|".join(_ORDINALS) + ")"), re.IGNORECASE)


def _compound_pattern() -> re.Pattern[str]:
    return re.compile(
        _WORD_BOUNDARY.format("(" + "|".join(_TENS) + r")[ -]+(" + "|".join(_SMALL) + ")"),
        re.IGNORECASE,
    )


def _single_pattern() -> re.Pattern[str]:
    # Exclude "one" initially: it is frequently an article/pronoun rather than a numeric expression.
    words = [word for word in list(_SMALL) + list(_TENS) if word != "one"]
    return re.compile(_WORD_BOUNDARY.format("(" + "|".join(words) + ")"), re.IGNORECASE)


def build_number_variant(text: str) -> str | None:
    """Return one conservative digit-form variant, or None if no safe rewrite exists."""
    if re.search(r"\d", text):
        return None
    # Large-number constructions have pronunciation ambiguity; leave them untouched.
    if re.search(r"\b(?:hundred|thousand|million|billion)\b", text, re.IGNORECASE):
        return None

    time_pattern = _time_pattern()
    time_match = time_pattern.search(text)
    if time_match:
        minute = _TENS[time_match.group("minute_tens").lower()] + _SMALL[time_match.group("minute_small").lower()]
        if minute > 59:
            return None
        return time_pattern.sub(_clock_replacement, text, count=1)

    ordinal_pattern = _ordinal_pattern()
    ordinal_match = ordinal_pattern.search(text)
    if ordinal_match:
        word = ordinal_match.group(1).lower()
        replacement = _ORDINALS[word][1]
        return text[: ordinal_match.start()] + replacement + text[ordinal_match.end() :]

    compound_pattern = _compound_pattern()
    compound_match = compound_pattern.search(text)
    if compound_match:
        value = _TENS[compound_match.group(1).lower()] + _SMALL[compound_match.group(2).lower()]
        return text[: compound_match.start()] + str(value) + text[compound_match.end() :]

    single_pattern = _single_pattern()
    single_match = single_pattern.search(text)
    if single_match:
        word = single_match.group(1).lower()
        value = _SMALL.get(word, _TENS.get(word))
        return text[: single_match.start()] + str(value) + text[single_match.end() :]
    return None


def _replace_prompt(record: dict, prompt: str) -> dict:
    result = copy.deepcopy(record)
    result["prompt"] = prompt
    return result


def make_focused_dataset(
    records: Iterable[dict],
    pair_limit: int,
    replay_count: int,
    eval_count: int,
    seed: int = 42,
) -> dict:
    """Create source-disjoint focused train/eval records from in-memory MOSS records."""
    rng = random.Random(seed)
    records = list(records)
    candidates = []
    for record in records:
        variant = build_number_variant(record["prompt"])
        if variant and variant != record["prompt"]:
            candidates.append((record, variant, classify_safe_variant(record["prompt"])))
    rng.shuffle(candidates)

    eval_candidates = candidates[:eval_count]
    eval_sources = {record.get("source_path") for record, _, _ in eval_candidates}
    train_candidates = [item for item in candidates[eval_count:] if item[0].get("source_path") not in eval_sources]
    train_candidates = train_candidates[:pair_limit]
    pair_sources = {record.get("source_path") for record, _, _ in train_candidates}

    pairs = []
    train = []
    for record, variant, category in train_candidates:
        word_record = copy.deepcopy(record)
        digit_record = _replace_prompt(record, variant)
        pairs.append({"word_record": word_record, "digit_record": digit_record, "category": category})
        train.extend([word_record, digit_record])

    replay_pool = [
        record for record in records
        if record.get("source_path") not in eval_sources | pair_sources
    ]
    rng.shuffle(replay_pool)
    train.extend(copy.deepcopy(replay_pool[:replay_count]))
    rng.shuffle(train)

    evaluation = []
    for record, variant, category in eval_candidates:
        evaluation.append({
            "source_path": record.get("source_path"),
            "word_prompt": record["prompt"],
            "digit_prompt": variant,
            "category": category,
        })
    return {"train": train, "eval": evaluation, "pairs": pairs}
