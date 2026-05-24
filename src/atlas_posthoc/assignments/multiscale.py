from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal, Mapping

from atlas_posthoc.io.atlas_reader import canonicalize_unit_key


MissingPolicy = Literal["error", "drop", "none"]
AmbiguousPolicy = Literal["error", "coarsest", "finest"]


class MultiscaleAssignmentError(Exception):
    """Base exception for multiscale assignment expansion errors."""


class MissingAssignmentError(MultiscaleAssignmentError):
    """Raised when a node cannot be matched to any multiscale assignment key."""


class AmbiguousAssignmentError(MultiscaleAssignmentError):
    """Raised when a node matches multiple assignment keys."""


@dataclass
class MultiscaleExpansionSummary:
    n_nodes: int = 0
    n_assigned: int = 0
    n_missing: int = 0
    n_ambiguous: int = 0

    key_cols: list[str] = field(default_factory=list)
    node_id_col: str | None = None

    used_assignment_key_counts: dict[tuple[str, ...], int] = field(default_factory=dict)
    unmatched_assignment_keys: list[tuple[str, ...]] = field(default_factory=list)

    missing_nodes: list[Any] = field(default_factory=list)
    ambiguous_nodes: list[Any] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_error(self, message: str, max_errors: int = 20) -> None:
        if len(self.errors) < max_errors:
            self.errors.append(message)

    def to_dict(self) -> dict[str, Any]:
        """
        JSON-friendly summary dictionary.
        """
        out = asdict(self)

        out["used_assignment_key_counts"] = {
            key_to_string(k): v for k, v in self.used_assignment_key_counts.items()
        }
        out["unmatched_assignment_keys"] = [
            key_to_string(k) for k in self.unmatched_assignment_keys
        ]

        return out


@dataclass
class MultiscaleExpansionResult:
    node_to_district: dict[Any, int | None]
    node_to_assignment_key: dict[Any, tuple[str, ...] | None]
    summary: MultiscaleExpansionSummary


def key_to_string(key: tuple[str, ...]) -> str:
    """
    Stable printable representation for hierarchical assignment keys.
    """
    return "[" + ", ".join(repr(x) for x in key) + "]"


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    return False


def _coerce_district_label(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean values are not valid district labels")

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError(f"non-integer float district label: {value!r}")

    if isinstance(value, str):
        return int(value.strip())

    raise ValueError(f"cannot convert district label to int: {value!r}")


def normalize_multiscale_assignment(
    assignment: Mapping[Any, Any],
) -> dict[tuple[str, ...], int]:
    """
    Normalize assignment keys and district labels.

    Parameters
    ----------
    assignment:
        Mapping from raw or canonical AtlasIO keys to district labels.

    Returns
    -------
    dict
        Mapping from tuple keys to integer district labels.

    Examples
    --------
    >>> normalize_multiscale_assignment({'["FRANKLIN", "001"]': '3'})
    {('FRANKLIN', '001'): 3}
    """
    out: dict[tuple[str, ...], int] = {}

    for raw_key, raw_value in assignment.items():
        key = canonicalize_unit_key(raw_key)
        value = _coerce_district_label(raw_value)

        if key in out and out[key] != value:
            raise MultiscaleAssignmentError(
                f"canonical key collision for {key!r}: {out[key]!r} vs {value!r}"
            )

        out[key] = value

    return out


def _lookup_node_field(
    node: Mapping[str, Any],
    field: str,
    *,
    case_insensitive: bool = False,
) -> Any:
    if field in node:
        return node[field]

    if case_insensitive:
        target = field.lower()
        for k, v in node.items():
            if str(k).lower() == target:
                return v

    raise KeyError(f"node is missing required field {field!r}")


def build_node_key(
    node: Mapping[str, Any],
    key_cols: Iterable[str],
    *,
    case_insensitive_cols: bool = False,
) -> tuple[str, ...]:
    """
    Build the full finest-level hierarchical key for a node.

    For example, with key_cols=["county", "prec_id"], a node like

        {"county": "FRANKLIN", "prec_id": "001"}

    becomes

        ("FRANKLIN", "001")
    """
    values: list[str] = []

    for col in key_cols:
        value = _lookup_node_field(
            node,
            col,
            case_insensitive=case_insensitive_cols,
        )

        if _is_missing_value(value):
            raise KeyError(f"node field {col!r} is missing/null")

        values.append(str(value))

    return tuple(values)


def get_node_id(
    node: Mapping[str, Any],
    *,
    index: int,
    node_id_col: str | None = "id",
    case_insensitive_cols: bool = False,
) -> Any:
    """
    Get a stable node id from node metadata.

    If node_id_col is None, use the node's position in the input sequence.
    """
    if node_id_col is None:
        return index

    try:
        value = _lookup_node_field(
            node,
            node_id_col,
            case_insensitive=case_insensitive_cols,
        )
    except KeyError:
        return index

    if _is_missing_value(value):
        return index

    return value


def key_prefixes(key: tuple[str, ...]) -> list[tuple[str, ...]]:
    """
    Return all nonempty prefixes of a hierarchical key.

    Example
    -------
    >>> key_prefixes(("A", "B", "C"))
    [('A',), ('A', 'B'), ('A', 'B', 'C')]
    """
    return [key[:i] for i in range(1, len(key) + 1)]


def matching_assignment_keys(
    node_key: tuple[str, ...],
    assignment: Mapping[tuple[str, ...], int],
) -> list[tuple[str, ...]]:
    """
    Find all assignment keys that cover a finest-level node key.

    A key covers a node if it is a prefix of the node's finest-level key.
    """
    return [prefix for prefix in key_prefixes(node_key) if prefix in assignment]


def resolve_node_assignment(
    node_key: tuple[str, ...],
    assignment: Mapping[tuple[str, ...], int],
    *,
    ambiguous_policy: AmbiguousPolicy = "error",
) -> tuple[int, tuple[str, ...]]:
    """
    Resolve the district assignment for one node key.

    In valid multiscale Atlas output, each finest-level node should match exactly
    one assignment key. For example, either:

        ("FRANKLIN",)

    or:

        ("FRANKLIN", "001")

    should cover a Franklin precinct, but not both.

    If multiple keys match, the default is to raise an error. This is safer than
    silently choosing a coarser or finer key.
    """
    matches = matching_assignment_keys(node_key, assignment)

    if not matches:
        raise MissingAssignmentError(
            f"no assignment key covers node key {node_key!r}"
        )

    if len(matches) > 1:
        if ambiguous_policy == "error":
            raise AmbiguousAssignmentError(
                f"node key {node_key!r} is covered by multiple assignment keys: {matches!r}"
            )

        if ambiguous_policy == "coarsest":
            chosen = matches[0]
        elif ambiguous_policy == "finest":
            chosen = matches[-1]
        else:
            raise ValueError(f"unknown ambiguous_policy: {ambiguous_policy!r}")
    else:
        chosen = matches[0]

    return assignment[chosen], chosen


def expand_multiscale_assignment(
    assignment: Mapping[Any, Any],
    nodes: Iterable[Mapping[str, Any]],
    key_cols: Iterable[str],
    *,
    node_id_col: str | None = "id",
    missing_policy: MissingPolicy = "error",
    ambiguous_policy: AmbiguousPolicy = "error",
    case_insensitive_cols: bool = False,
) -> MultiscaleExpansionResult:
    """
    Expand a multiscale Atlas assignment to node-level assignments.

    Parameters
    ----------
    assignment:
        Multiscale assignment from AtlasIO, usually `record.assignment`.

        Keys may already be tuple keys, or raw AtlasIO JSON-array strings.

    nodes:
        Iterable of node metadata dictionaries. Each node must contain the fields
        named by `key_cols`.

    key_cols:
        Hierarchical columns from coarsest to finest level.

        Example:

            ["county", "prec_id"]

        A node with county="FRANKLIN", prec_id="001" has full key:

            ("FRANKLIN", "001")

        It can be covered by either:

            ("FRANKLIN",)

        or:

            ("FRANKLIN", "001")

    node_id_col:
        Field used as the output node id. If missing, the input index is used.
        If None, always use the input index.

    missing_policy:
        What to do when a node is not covered by any assignment key.

        - "error": raise MissingAssignmentError.
        - "drop": omit the node from the output assignment.
        - "none": include the node with value None.

    ambiguous_policy:
        What to do when multiple assignment keys cover the same node.

        - "error": raise AmbiguousAssignmentError.
        - "coarsest": use the shortest matching prefix.
        - "finest": use the longest matching prefix.

    case_insensitive_cols:
        Whether to match node metadata columns case-insensitively.

    Returns
    -------
    MultiscaleExpansionResult
        Contains node_to_district, node_to_assignment_key, and a summary.

    Notes
    -----
    This function intentionally does not parse AtlasIO files. It assumes the raw
    AtlasIO record has already been parsed by `io.atlas_reader`.
    """
    key_cols_list = list(key_cols)

    if not key_cols_list:
        raise ValueError("key_cols must contain at least one column name.")

    if missing_policy not in {"error", "drop", "none"}:
        raise ValueError(f"unknown missing_policy: {missing_policy!r}")

    if ambiguous_policy not in {"error", "coarsest", "finest"}:
        raise ValueError(f"unknown ambiguous_policy: {ambiguous_policy!r}")

    normalized_assignment = normalize_multiscale_assignment(assignment)

    node_to_district: dict[Any, int | None] = {}
    node_to_assignment_key: dict[Any, tuple[str, ...] | None] = {}
    used_keys: Counter[tuple[str, ...]] = Counter()

    summary = MultiscaleExpansionSummary(
        key_cols=key_cols_list,
        node_id_col=node_id_col,
    )

    for index, node in enumerate(nodes):
        summary.n_nodes += 1

        node_id = get_node_id(
            node,
            index=index,
            node_id_col=node_id_col,
            case_insensitive_cols=case_insensitive_cols,
        )

        try:
            node_key = build_node_key(
                node,
                key_cols_list,
                case_insensitive_cols=case_insensitive_cols,
            )

            district, used_key = resolve_node_assignment(
                node_key,
                normalized_assignment,
                ambiguous_policy=ambiguous_policy,
            )

        except MissingAssignmentError as exc:
            summary.n_missing += 1
            summary.missing_nodes.append(node_id)
            summary.add_error(str(exc))

            if missing_policy == "error":
                raise

            if missing_policy == "none":
                node_to_district[node_id] = None
                node_to_assignment_key[node_id] = None

            continue

        except AmbiguousAssignmentError as exc:
            summary.n_ambiguous += 1
            summary.ambiguous_nodes.append(node_id)
            summary.add_error(str(exc))
            raise

        except KeyError as exc:
            summary.n_missing += 1
            summary.missing_nodes.append(node_id)
            summary.add_error(f"node {node_id!r}: {exc}")

            if missing_policy == "error":
                raise MissingAssignmentError(f"node {node_id!r}: {exc}") from exc

            if missing_policy == "none":
                node_to_district[node_id] = None
                node_to_assignment_key[node_id] = None

            continue

        node_to_district[node_id] = district
        node_to_assignment_key[node_id] = used_key
        used_keys[used_key] += 1
        summary.n_assigned += 1

    summary.used_assignment_key_counts = dict(used_keys)
    summary.unmatched_assignment_keys = sorted(
        [key for key in normalized_assignment if key not in used_keys],
        key=lambda x: (len(x), x),
    )

    return MultiscaleExpansionResult(
        node_to_district=node_to_district,
        node_to_assignment_key=node_to_assignment_key,
        summary=summary,
    )


def expand_record_assignment(
    record: Any,
    nodes: Iterable[Mapping[str, Any]],
    key_cols: Iterable[str],
    *,
    node_id_col: str | None = "id",
    missing_policy: MissingPolicy = "error",
    ambiguous_policy: AmbiguousPolicy = "error",
    case_insensitive_cols: bool = False,
) -> MultiscaleExpansionResult:
    """
    Convenience wrapper for AtlasRecord-like objects with an `.assignment` field.
    """
    if not hasattr(record, "assignment"):
        raise TypeError("record must have an `.assignment` attribute")

    return expand_multiscale_assignment(
        record.assignment,
        nodes,
        key_cols,
        node_id_col=node_id_col,
        missing_policy=missing_policy,
        ambiguous_policy=ambiguous_policy,
        case_insensitive_cols=case_insensitive_cols,
    )