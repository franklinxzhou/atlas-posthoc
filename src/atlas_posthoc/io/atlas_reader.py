from __future__ import annotations

import gzip
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, TextIO


class AtlasReaderError(Exception):
    """Base exception for AtlasIO parsing errors."""


class AtlasHeaderError(AtlasReaderError):
    """Raised when an Atlas header cannot be parsed."""


class AtlasRecordError(AtlasReaderError):
    """Raised when an Atlas map record cannot be parsed."""


@dataclass
class AtlasHeader:
    path: Path
    magic: Any | None = None
    description: str | None = None
    date: str | None = None
    atlas_param_type: str | None = None
    map_param_type: str | None = None
    atlas_param: dict[str, Any] = field(default_factory=dict)
    raw_header: dict[str, Any] | None = None
    raw_param: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["path"] = str(self.path)
        return out


@dataclass
class AtlasRecord:
    chain_id: str
    path: Path
    line_number: int
    map_index: int
    name: str | None
    step: int | None
    weight: int | float | None
    data: dict[str, Any]
    assignment: dict[tuple[str, ...], int]
    raw_assignment: dict[Any, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["path"] = str(self.path)
        return out


@dataclass
class ParseSummary:
    path: Path
    chain_id: str
    lines_seen: int = 0
    header_lines: int = 0
    records_seen: int = 0
    records_yielded: int = 0
    empty_lines: int = 0
    skipped_by_thinning: int = 0
    bad_json_lines: int = 0
    bad_record_lines: int = 0
    bad_assignment_lines: int = 0
    errors: list[str] = field(default_factory=list)

    def add_error(self, line_number: int, message: str, max_errors: int = 10) -> None:
        if len(self.errors) < max_errors:
            self.errors.append(f"line {line_number}: {message}")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["path"] = str(self.path)
        return out


def infer_chain_id(path: str | Path) -> str:
    """Infer a readable chain id from a filename."""
    p = Path(path)
    name = p.name

    for suffix in (".jsonl.gz", ".json.gz", ".gz", ".jsonl", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]

    return p.stem


def open_text_auto(path: str | Path) -> TextIO:
    """
    Open a plain text or gzip-compressed AtlasIO file.

    Supported:
        - .jsonl
        - .json
        - .txt
        - .jsonl.gz
        - .gz

    Not currently supported:
        - .bz2
    """
    p = Path(path)

    if p.suffix == ".bz2":
        raise NotImplementedError(
            "`.bz2` Atlas files are not supported yet. "
            "This package currently supports plain text and `.gz` files."
        )

    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8")

    return open(p, "rt", encoding="utf-8")


def parse_json_line(line: str, *, allow_double_encoded: bool = True) -> Any:
    """
    Parse one JSON line.

    Atlas outputs occasionally contain JSON strings that themselves contain JSON.
    If `allow_double_encoded=True`, this function tries a second json.loads when
    the first parse returns a string.
    """
    text = line.strip()
    if not text:
        raise json.JSONDecodeError("empty line", line, 0)

    obj = json.loads(text)

    if allow_double_encoded and isinstance(obj, str):
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            return obj

    return obj


def _parse_header_line(line: str) -> Any:
    """
    Parse a header line.

    Unlike map records, the first Atlas line may be a JSON string or a raw text
    magic line. If JSON parsing fails, return the stripped raw text.
    """
    text = line.strip()
    if not text:
        return None

    try:
        return parse_json_line(text)
    except json.JSONDecodeError:
        return text


def read_atlas_header(path: str | Path) -> AtlasHeader:
    """
    Read the first three AtlasIO header lines.

    Jonathan's PythonReader assumes:
        line 1: magic/description line
        line 2: atlas header dictionary
        line 3: atlas parameter dictionary

    This function follows that convention but is more tolerant of missing fields.
    """
    p = Path(path)

    with open_text_auto(p) as fp:
        line1 = fp.readline()
        line2 = fp.readline()
        line3 = fp.readline()

    if not line1 or not line2 or not line3:
        raise AtlasHeaderError(
            f"Atlas file {p} has fewer than three header lines."
        )

    magic = _parse_header_line(line1)
    raw_header_obj = _parse_header_line(line2)
    raw_param_obj = _parse_header_line(line3)

    raw_header = raw_header_obj if isinstance(raw_header_obj, dict) else None
    raw_param = raw_param_obj if isinstance(raw_param_obj, dict) else None

    header = AtlasHeader(path=p, magic=magic)

    if raw_header is not None:
        header.raw_header = raw_header
        header.description = raw_header.get("description")
        header.date = raw_header.get("date")
        header.atlas_param_type = raw_header.get("atlasParamType")
        header.map_param_type = raw_header.get("mapParamType")

    if raw_param is not None:
        header.raw_param = raw_param
        header.atlas_param = raw_param

    return header


_STEP_RE = re.compile(r"step\s*[-_:]?\s*(\d+)", re.IGNORECASE)


def parse_step(name: str | None) -> int | None:
    """
    Parse names like `step0`, `step10000`, or `step-10000`.

    Returns None when the name does not encode a step.
    """
    if name is None:
        return None

    match = _STEP_RE.search(str(name))
    if match is None:
        return None

    return int(match.group(1))


def flatten_districting(districting: Any) -> dict[Any, Any]:
    """
    Flatten an AtlasIO districting object.

    AtlasIO often stores districting as a list of singleton dictionaries:

        [
            {"[\"FRANKLIN\", \"001\"]": 3},
            {"[\"FRANKLIN\", \"002\"]": 3}
        ]

    This function turns it into one dictionary. If the input is already a
    dictionary, it returns a shallow copy.
    """
    if isinstance(districting, dict):
        return dict(districting)

    if not isinstance(districting, list):
        raise AtlasRecordError(
            f"`districting` must be a dict or a list of dicts; got {type(districting).__name__}."
        )

    out: dict[Any, Any] = {}

    for i, item in enumerate(districting):
        if not isinstance(item, dict):
            raise AtlasRecordError(
                f"`districting` list item {i} is not a dict; got {type(item).__name__}."
            )

        for key, value in item.items():
            if key in out and out[key] != value:
                raise AtlasRecordError(
                    f"duplicate districting key with conflicting values: {key!r}"
                )
            out[key] = value

    return out


def canonicalize_unit_key(key: Any) -> tuple[str, ...]:
    """
    Convert AtlasIO hierarchical keys into tuple keys.

    Examples:
        '["FRANKLIN", "001"]' -> ("FRANKLIN", "001")
        '["FRANKLIN"]'        -> ("FRANKLIN",)
        ["A", "B", "C"]       -> ("A", "B", "C")
        ("A", "B")            -> ("A", "B")

    If a string is not JSON-decodable, it is treated as a one-level key.
    """
    if isinstance(key, tuple):
        return tuple(str(x) for x in key)

    if isinstance(key, list):
        return tuple(str(x) for x in key)

    if isinstance(key, str):
        text = key.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return (text,)

        # Handle the rare case where the key is double-encoded.
        if isinstance(parsed, str):
            try:
                parsed2 = json.loads(parsed)
            except json.JSONDecodeError:
                return (parsed,)
            parsed = parsed2

        if isinstance(parsed, list):
            return tuple(str(x) for x in parsed)

        if isinstance(parsed, tuple):
            return tuple(str(x) for x in parsed)

        return (str(parsed),)

    return (str(key),)


def _coerce_district_label(value: Any) -> int:
    """
    Convert a district label to int.

    AtlasIO district labels are expected to be integer-like. We reject bools
    because bool is a subclass of int in Python and should not be a district id.
    """
    if isinstance(value, bool):
        raise ValueError("boolean values are not valid district labels")

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError(f"non-integer float district label: {value!r}")

    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"non-integer string district label: {value!r}") from exc

    raise ValueError(f"cannot convert district label to int: {value!r}")


def normalize_assignment(districting: Any) -> tuple[dict[tuple[str, ...], int], dict[Any, Any]]:
    """
    Flatten and canonicalize a raw AtlasIO districting object.

    Returns:
        assignment:
            dict mapping canonical tuple unit keys to integer district labels.

        raw_assignment:
            flattened raw assignment, preserving original AtlasIO keys.
    """
    raw_assignment = flatten_districting(districting)
    assignment: dict[tuple[str, ...], int] = {}

    for raw_key, raw_value in raw_assignment.items():
        key = canonicalize_unit_key(raw_key)
        value = _coerce_district_label(raw_value)

        if key in assignment and assignment[key] != value:
            raise AtlasRecordError(
                f"canonical key collision for {key!r}: "
                f"{assignment[key]!r} vs {value!r}"
            )

        assignment[key] = value

    return assignment, raw_assignment


def _looks_like_bare_assignment(obj: dict[Any, Any]) -> bool:
    """
    Heuristic for nonstandard files where the whole JSON object is the assignment.

    This is intentionally conservative. Official AtlasIO map records should contain
    a `districting` field.
    """
    if not obj:
        return False

    for value in obj.values():
        try:
            _coerce_district_label(value)
        except ValueError:
            return False

    return True


def _record_from_object(
    obj: dict[str, Any],
    *,
    path: Path,
    chain_id: str,
    line_number: int,
    map_index: int,
) -> AtlasRecord:
    if "districting" in obj:
        districting_obj = obj["districting"]
        name = obj.get("name")
        weight = obj.get("weight")
        raw_data = obj.get("data", {})
    elif _looks_like_bare_assignment(obj):
        # Nonstandard fallback for files that store one assignment per line.
        districting_obj = obj
        name = None
        weight = None
        raw_data = {}
    else:
        raise AtlasRecordError(
            "map record does not contain a `districting` field "
            "and does not look like a bare assignment dictionary"
        )

    assignment, raw_assignment = normalize_assignment(districting_obj)

    if raw_data is None:
        data: dict[str, Any] = {}
    elif isinstance(raw_data, dict):
        data = raw_data
    else:
        data = {"_raw_data": raw_data}

    return AtlasRecord(
        chain_id=chain_id,
        path=path,
        line_number=line_number,
        map_index=map_index,
        name=str(name) if name is not None else None,
        step=parse_step(str(name) if name is not None else None),
        weight=weight,
        data=data,
        assignment=assignment,
        raw_assignment=raw_assignment,
    )


def iter_atlas_records(
    path: str | Path,
    *,
    chain_id: str | None = None,
    thin_step: int = 1,
    skip_bad_lines: bool = True,
    summary: ParseSummary | None = None,
) -> Iterator[AtlasRecord]:
    """
    Iterate over valid AtlasIO map records.

    Parameters
    ----------
    path:
        AtlasIO output file. Supports plain text and `.gz`.

    chain_id:
        Optional chain label. If omitted, inferred from filename.

    thin_step:
        Keep every `thin_step`-th map line after the three-line header.

    skip_bad_lines:
        If True, malformed records are counted and skipped. If False, the first
        malformed line raises an exception.

    summary:
        Optional ParseSummary object to mutate during parsing.

    Notes
    -----
    Thinning is applied to map lines after the three-line Atlas header, using
    zero-based map-line indexing. Header lines are never yielded.
    """
    p = Path(path)

    if thin_step <= 0:
        raise ValueError("thin_step must be a positive integer.")

    cid = chain_id or infer_chain_id(p)

    if summary is None:
        summary = ParseSummary(path=p, chain_id=cid)

    yielded_index = 0

    with open_text_auto(p) as fp:
        for line_number, line in enumerate(fp, start=1):
            summary.lines_seen += 1

            # AtlasIO convention: first three lines are header/metadata.
            if line_number <= 3:
                summary.header_lines += 1
                continue

            if not line.strip():
                summary.empty_lines += 1
                continue

            map_line_index = line_number - 4

            if map_line_index % thin_step != 0:
                summary.skipped_by_thinning += 1
                continue

            summary.records_seen += 1

            try:
                obj = parse_json_line(line)
            except json.JSONDecodeError as exc:
                summary.bad_json_lines += 1
                summary.add_error(line_number, f"bad JSON: {exc}")
                if skip_bad_lines:
                    continue
                raise AtlasRecordError(
                    f"bad JSON in {p} at line {line_number}: {exc}"
                ) from exc

            if not isinstance(obj, dict):
                summary.bad_record_lines += 1
                summary.add_error(
                    line_number,
                    f"expected map record dict, got {type(obj).__name__}",
                )
                if skip_bad_lines:
                    continue
                raise AtlasRecordError(
                    f"expected map record dict in {p} at line {line_number}, "
                    f"got {type(obj).__name__}"
                )

            try:
                record = _record_from_object(
                    obj,
                    path=p,
                    chain_id=cid,
                    line_number=line_number,
                    map_index=yielded_index,
                )
            except AtlasRecordError as exc:
                message = str(exc)
                if "districting" in message or "assignment" in message or "label" in message:
                    summary.bad_assignment_lines += 1
                else:
                    summary.bad_record_lines += 1

                summary.add_error(line_number, message)

                if skip_bad_lines:
                    continue

                raise

            summary.records_yielded += 1
            yielded_index += 1
            yield record


def read_atlas_records(
    path: str | Path,
    *,
    chain_id: str | None = None,
    thin_step: int = 1,
    skip_bad_lines: bool = True,
) -> tuple[AtlasHeader, list[AtlasRecord], ParseSummary]:
    """
    Convenience function for reading a whole Atlas file into memory.

    This is useful for tests and small files. For production chain files, prefer
    `iter_atlas_records`.
    """
    p = Path(path)
    cid = chain_id or infer_chain_id(p)
    header = read_atlas_header(p)
    summary = ParseSummary(path=p, chain_id=cid)

    records = list(
        iter_atlas_records(
            p,
            chain_id=cid,
            thin_step=thin_step,
            skip_bad_lines=skip_bad_lines,
            summary=summary,
        )
    )

    return header, records, summary