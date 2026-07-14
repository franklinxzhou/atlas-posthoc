from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from atlas_posthoc.assignments.multiscale import (
    AmbiguousAssignmentError,
    MissingAssignmentError,
    build_node_key,
    matching_assignment_keys,
    normalize_multiscale_assignment,
)


class AssignmentKeyError(Exception):
    """Base exception for assignment-key diagnostics."""


@dataclass
class KeySchemeDiagnostic:
    key_cols: list[str]
    n_nodes: int
    n_assigned: int
    n_missing: int
    n_ambiguous: int
    n_assignment_keys: int
    n_used_assignment_keys: int
    n_unmatched_assignment_keys: int
    example_missing_nodes: list[Any]
    example_ambiguous_nodes: list[Any]
    example_unmatched_assignment_keys: list[tuple[str, ...]]

    @property
    def assigned_fraction(self) -> float:
        return self.n_assigned / self.n_nodes if self.n_nodes else 0.0

    @property
    def is_perfect_match(self) -> bool:
        return (
            self.n_nodes > 0
            and self.n_missing == 0
            and self.n_ambiguous == 0
            and self.n_unmatched_assignment_keys == 0
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["assigned_fraction"] = self.assigned_fraction
        out["is_perfect_match"] = self.is_perfect_match
        out["example_unmatched_assignment_keys"] = [
            list(k) for k in self.example_unmatched_assignment_keys
        ]
        return out


def default_candidate_key_schemes(
    nodes: Iterable[Mapping[str, Any]] | None = None,
) -> list[list[str]]:
    """
    Candidate key schemes useful for Ohio-style graph files.

    The baseline Ohio graph uses scalar ``MCD`` while the comparison graph uses
    scalar ``COUSUB``. Municipality candidates are included only when the
    selected graph has that scalar column. ``PLACE`` is deliberately never an
    assignment-key candidate because it is a population mapping, not node
    identity.

    AtlasIO outputs may use either coarse county-level keys or finer graph-node
    keys. We test several plausible schemes and let the diagnostic decide.
    """
    if nodes is None:
        municipality_cols = ["MCD"]
    else:
        nodes_list = list(nodes)
        municipality_cols = [
            col
            for col in ("MCD", "COUSUB")
            if nodes_list and all(col in node for node in nodes_list)
        ]

    candidates = [
        ["COUNTY", "NAME"],
        ["COUNTY", "ORIG_NAME"],
    ]
    candidates.extend([["COUNTY", col] for col in municipality_cols])
    candidates.extend([["NAME"], ["ORIG_NAME"]])
    candidates.extend([[col] for col in municipality_cols])
    candidates.append(["COUNTY"])
    return candidates


def node_id_from_node(
    node: Mapping[str, Any],
    *,
    index: int,
    node_id_col: str | None = "id",
) -> Any:
    if node_id_col is None:
        return index

    return node.get(node_id_col, index)


def diagnose_key_scheme(
    assignment: Mapping[Any, Any],
    nodes: Iterable[Mapping[str, Any]],
    key_cols: Sequence[str],
    *,
    node_id_col: str | None = "id",
    max_examples: int = 5,
    case_insensitive_cols: bool = False,
) -> KeySchemeDiagnostic:
    """
    Diagnose whether one candidate key scheme matches an AtlasIO assignment.

    Parameters
    ----------
    assignment:
        AtlasIO assignment, usually `record.assignment`.

    nodes:
        Graph node metadata.

    key_cols:
        Candidate hierarchy columns, for example:
            ["COUNTY", "NAME"]
            ["NAME"]
            ["COUNTY"]

    Returns
    -------
    KeySchemeDiagnostic
        Counts of assigned, missing, ambiguous, and unmatched keys.
    """
    key_cols_list = list(key_cols)
    if not key_cols_list:
        raise AssignmentKeyError("key_cols must contain at least one column")

    normalized = normalize_multiscale_assignment(assignment)
    nodes_list = list(nodes)

    n_assigned = 0
    n_missing = 0
    n_ambiguous = 0

    used_keys: set[tuple[str, ...]] = set()
    missing_nodes: list[Any] = []
    ambiguous_nodes: list[Any] = []

    for index, node in enumerate(nodes_list):
        node_id = node_id_from_node(node, index=index, node_id_col=node_id_col)

        try:
            node_key = build_node_key(
                node,
                key_cols_list,
                case_insensitive_cols=case_insensitive_cols,
            )
        except KeyError:
            n_missing += 1
            if len(missing_nodes) < max_examples:
                missing_nodes.append(node_id)
            continue

        matches = matching_assignment_keys(node_key, normalized)

        if not matches:
            n_missing += 1
            if len(missing_nodes) < max_examples:
                missing_nodes.append(node_id)
            continue

        if len(matches) > 1:
            n_ambiguous += 1
            if len(ambiguous_nodes) < max_examples:
                ambiguous_nodes.append(node_id)
            continue

        n_assigned += 1
        used_keys.add(matches[0])

    unmatched_keys = sorted(
        [key for key in normalized if key not in used_keys],
        key=lambda x: (len(x), x),
    )

    return KeySchemeDiagnostic(
        key_cols=key_cols_list,
        n_nodes=len(nodes_list),
        n_assigned=n_assigned,
        n_missing=n_missing,
        n_ambiguous=n_ambiguous,
        n_assignment_keys=len(normalized),
        n_used_assignment_keys=len(used_keys),
        n_unmatched_assignment_keys=len(unmatched_keys),
        example_missing_nodes=missing_nodes,
        example_ambiguous_nodes=ambiguous_nodes,
        example_unmatched_assignment_keys=unmatched_keys[:max_examples],
    )


def diagnose_key_schemes(
    assignment: Mapping[Any, Any],
    nodes: Iterable[Mapping[str, Any]],
    candidate_key_cols: Iterable[Sequence[str]] | None = None,
    *,
    node_id_col: str | None = "id",
    max_examples: int = 5,
    case_insensitive_cols: bool = False,
) -> list[KeySchemeDiagnostic]:
    """
    Diagnose multiple candidate key schemes.

    Results are sorted with the best schemes first:
      1. most assigned nodes;
      2. fewest ambiguous nodes;
      3. fewest unmatched assignment keys;
      4. shortest key scheme.
    """
    nodes_list = list(nodes)

    if candidate_key_cols is None:
        candidate_key_cols = default_candidate_key_schemes(nodes_list)

    diagnostics = [
        diagnose_key_scheme(
            assignment,
            nodes_list,
            key_cols,
            node_id_col=node_id_col,
            max_examples=max_examples,
            case_insensitive_cols=case_insensitive_cols,
        )
        for key_cols in candidate_key_cols
    ]

    diagnostics.sort(
        key=lambda d: (
            -d.n_assigned,
            d.n_ambiguous,
            d.n_unmatched_assignment_keys,
            len(d.key_cols),
        )
    )

    return diagnostics


def choose_best_key_scheme(
    assignment: Mapping[Any, Any],
    nodes: Iterable[Mapping[str, Any]],
    candidate_key_cols: Iterable[Sequence[str]] | None = None,
    *,
    node_id_col: str | None = "id",
    require_perfect: bool = True,
    case_insensitive_cols: bool = False,
) -> KeySchemeDiagnostic:
    """
    Choose the best key scheme from diagnostics.

    If `require_perfect=True`, this raises unless the best scheme assigns every
    node with no ambiguity and no unmatched assignment keys.
    """
    diagnostics = diagnose_key_schemes(
        assignment,
        nodes,
        candidate_key_cols,
        node_id_col=node_id_col,
        case_insensitive_cols=case_insensitive_cols,
    )

    if not diagnostics:
        raise AssignmentKeyError("no candidate key schemes were provided")

    best = diagnostics[0]

    if require_perfect and not best.is_perfect_match:
        raise AssignmentKeyError(
            "no perfect key scheme found. "
            f"Best scheme {best.key_cols} assigned {best.n_assigned}/{best.n_nodes} nodes, "
            f"missing={best.n_missing}, ambiguous={best.n_ambiguous}, "
            f"unmatched_assignment_keys={best.n_unmatched_assignment_keys}."
        )

    return best


def format_key_scheme_diagnostics(
    diagnostics: Sequence[KeySchemeDiagnostic],
) -> str:
    """
    Format diagnostics as a small readable table.
    """
    lines = []
    lines.append(
        "key_cols | assigned | missing | ambiguous | used_keys | unmatched_keys | perfect"
    )
    lines.append(
        "--- | ---: | ---: | ---: | ---: | ---: | ---:"
    )

    for d in diagnostics:
        lines.append(
            f"{d.key_cols} | "
            f"{d.n_assigned}/{d.n_nodes} | "
            f"{d.n_missing} | "
            f"{d.n_ambiguous} | "
            f"{d.n_used_assignment_keys}/{d.n_assignment_keys} | "
            f"{d.n_unmatched_assignment_keys} | "
            f"{d.is_perfect_match}"
        )

    return "\n".join(lines)
