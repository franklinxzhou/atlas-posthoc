from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence


SortOrder = Literal["ascending", "descending", "none"]


class CompactnessError(Exception):
    """Base exception for compactness observable errors."""


@dataclass
class OrderedCompactnessResult:
    ordered_scores: list[float | None]
    raw_scores: list[float | None]
    source_key: str | None
    n_districts_expected: int | None
    n_districts_observed: int
    missing_score_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_COMPACTNESS_KEYS = (
    "get_isoperimetric_scores",
    "get_isoperimetric_score",
    "isoperimetric_scores",
    "isoperimetric_score",
    "compactness_scores",
    "compactness_score",
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    return False


def _as_float_or_none(value: Any) -> float | None:
    if _is_missing(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CompactnessError(f"could not convert compactness score {value!r} to float") from exc


def _coerce_score_sequence(value: Any) -> list[float | None]:
    """
    Convert a compactness payload into a list of scores.

    Accepts:
      - list/tuple of numbers;
      - dict mapping district labels to numbers;
      - scalar number, treated as a length-one vector.

    Dict values are sorted by district label when possible.
    """
    if isinstance(value, Mapping):
        try:
            items = sorted(value.items(), key=lambda kv: int(kv[0]))
        except (TypeError, ValueError):
            items = sorted(value.items(), key=lambda kv: str(kv[0]))

        return [_as_float_or_none(v) for _, v in items]

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_as_float_or_none(v) for v in value]

    return [_as_float_or_none(value)]


def find_compactness_payload(
    data: Mapping[str, Any],
    *,
    keys: Iterable[str] = DEFAULT_COMPACTNESS_KEYS,
) -> tuple[str, Any]:
    """
    Find the compactness payload inside an AtlasRecord `.data` dictionary.

    Returns
    -------
    (key, payload)

    Raises
    ------
    CompactnessError
        If none of the requested keys are present.
    """
    for key in keys:
        if key in data:
            return key, data[key]

    available = sorted(str(k) for k in data.keys())
    raise CompactnessError(
        "could not find compactness scores in record data. "
        f"Tried keys={list(keys)!r}. Available keys={available!r}."
    )


def extract_compactness_scores_from_data(
    data: Mapping[str, Any],
    *,
    keys: Iterable[str] = DEFAULT_COMPACTNESS_KEYS,
) -> tuple[list[float | None], str]:
    """
    Extract compactness scores from a record data dictionary.

    This is intended for AtlasIO records where compactness/isoperimetric scores
    were saved in the `data` field.
    """
    source_key, payload = find_compactness_payload(data, keys=keys)
    scores = _coerce_score_sequence(payload)
    return scores, source_key


def ordered_compactness_from_scores(
    scores: Iterable[Any],
    *,
    n_districts: int | None = None,
    sort: SortOrder = "ascending",
    allow_missing_scores: bool = False,
    source_key: str | None = None,
) -> OrderedCompactnessResult:
    """
    Convert raw district compactness scores into an ordered vector.

    For isoperimetric scores, ascending order means the first rank is the most
    compact district when lower is better.
    """
    raw_scores = [_as_float_or_none(x) for x in scores]

    missing_score_count = sum(x is None for x in raw_scores)

    if missing_score_count and not allow_missing_scores:
        raise CompactnessError(
            f"compactness vector contains {missing_score_count} missing scores"
        )

    if n_districts is not None and len(raw_scores) != n_districts:
        raise CompactnessError(
            f"expected {n_districts} compactness scores, got {len(raw_scores)}"
        )

    numeric_scores = [x for x in raw_scores if x is not None]
    none_tail: list[None] = [None] * missing_score_count

    if sort == "ascending":
        ordered: list[float | None] = sorted(numeric_scores) + none_tail
    elif sort == "descending":
        ordered = sorted(numeric_scores, reverse=True) + none_tail
    elif sort == "none":
        ordered = raw_scores
    else:
        raise ValueError(f"unknown sort order: {sort!r}")

    return OrderedCompactnessResult(
        ordered_scores=ordered,
        raw_scores=raw_scores,
        source_key=source_key,
        n_districts_expected=n_districts,
        n_districts_observed=len(raw_scores),
        missing_score_count=missing_score_count,
    )


def extract_ordered_compactness_from_data(
    data: Mapping[str, Any],
    *,
    keys: Iterable[str] = DEFAULT_COMPACTNESS_KEYS,
    n_districts: int | None = None,
    sort: SortOrder = "ascending",
    allow_missing_scores: bool = False,
) -> OrderedCompactnessResult:
    """
    Extract and order compactness scores from an AtlasIO record data dictionary.
    """
    scores, source_key = extract_compactness_scores_from_data(data, keys=keys)

    return ordered_compactness_from_scores(
        scores,
        n_districts=n_districts,
        sort=sort,
        allow_missing_scores=allow_missing_scores,
        source_key=source_key,
    )


def extract_ordered_compactness_from_record(
    record: Any,
    *,
    keys: Iterable[str] = DEFAULT_COMPACTNESS_KEYS,
    n_districts: int | None = None,
    sort: SortOrder = "ascending",
    allow_missing_scores: bool = False,
) -> OrderedCompactnessResult:
    """
    Convenience wrapper for AtlasRecord-like objects with a `.data` field.
    """
    if not hasattr(record, "data"):
        raise TypeError("record must have a `.data` attribute")

    data = record.data

    if not isinstance(data, Mapping):
        raise CompactnessError(
            f"record.data must be a mapping, got {type(data).__name__}"
        )

    return extract_ordered_compactness_from_data(
        data,
        keys=keys,
        n_districts=n_districts,
        sort=sort,
        allow_missing_scores=allow_missing_scores,
    )