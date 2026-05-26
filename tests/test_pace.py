import numpy as np
import pytest

from atlas_posthoc.diagnostics.pace import (
    PACEError,
    bootstrap_pace_delta,
    canonicalize_assignment_state,
    contingency_table,
    exact_state_labels_from_chain_states,
    exact_state_labels_from_matrix,
    pace_delta_from_contingency,
    pace_exact_from_chain_states,
    pace_from_contingency,
    pace_from_labels,
)


def test_contingency_table():
    labels = [0, 0, 1, 1, 1, 0]
    chains = ["a", "a", "a", "b", "b", "b"]

    table, chain_names, unique_labels = contingency_table(labels, chains)

    assert chain_names == ["a", "b"]
    assert unique_labels == [0, 1]
    assert table.tolist() == [
        [2, 1],
        [1, 2],
    ]


def test_pace_delta_identical_chain_distributions():
    table = np.array(
        [
            [5, 5],
            [5, 5],
            [5, 5],
            [5, 5],
        ]
    )

    delta, raw, correction = pace_delta_from_contingency(table)

    assert raw == 0.0
    assert delta == 0.0
    assert correction == pytest.approx(np.sqrt(4) / 3)


def test_pace_delta_separated_chains():
    table = np.array(
        [
            [10, 0],
            [0, 10],
        ]
    )

    delta, raw, correction = pace_delta_from_contingency(
        table,
        apply_correction=False,
    )

    # Pooled distribution is [0.5, 0.5].
    # Each chain has TV distance 0.5 from pooled.
    assert raw == pytest.approx(0.5)
    assert delta == pytest.approx(0.5)
    assert correction == 1.0


def test_pace_from_labels():
    labels = [0, 0, 1, 1, 1, 0]
    chains = ["a", "a", "a", "b", "b", "b"]

    result = pace_from_labels(
        labels,
        chains,
        threshold=0.5,
        apply_correction=False,
    )

    assert result.n_chains == 2
    assert result.n_cells == 2
    assert result.n_total_samples == 6
    assert result.chain_sizes == {"a": 3, "b": 3}
    assert result.delta == pytest.approx(1 / 6)
    assert result.converged is True


def test_pace_from_contingency_with_bootstrap():
    table = np.array(
        [
            [5, 5],
            [5, 5],
            [5, 5],
        ]
    )

    result = pace_from_contingency(
        table,
        threshold=0.5,
        n_bootstrap=20,
        random_state=123,
    )

    assert result.bootstrap is not None
    assert result.bootstrap.n_bootstrap == 20
    assert len(result.bootstrap.bootstrap_deltas) == 20
    assert result.converged is not None


def test_bootstrap_pace_delta_reproducible():
    table = np.array(
        [
            [5, 5],
            [5, 5],
        ]
    )

    b1 = bootstrap_pace_delta(table, n_bootstrap=10, random_state=1)
    b2 = bootstrap_pace_delta(table, n_bootstrap=10, random_state=1)

    assert b1.bootstrap_deltas == b2.bootstrap_deltas


def test_canonicalize_assignment_state_label_invariant():
    state_a = [10, 10, 20, 20, 30]
    state_b = [1, 1, 2, 2, 3]

    assert canonicalize_assignment_state(state_a) == canonicalize_assignment_state(state_b)


def test_exact_state_labels_from_matrix():
    states = np.array(
        [
            [1, 1, 2, 2],
            [9, 9, 8, 8],  # same after relabeling
            [1, 2, 1, 2],
        ]
    )

    labels, label_to_state = exact_state_labels_from_matrix(states, relabel=True)

    assert labels.tolist() == [0, 0, 1]
    assert len(label_to_state) == 2


def test_exact_state_labels_from_chain_states():
    chain_states = {
        "run1": np.array(
            [
                [1, 1, 2, 2],
                [1, 2, 1, 2],
            ]
        ),
        "run2": np.array(
            [
                [9, 9, 8, 8],
                [3, 4, 3, 4],
            ]
        ),
    }

    labels, chain_ids, label_to_state = exact_state_labels_from_chain_states(
        chain_states,
        relabel=True,
    )

    assert labels.tolist() == [0, 1, 0, 1]
    assert chain_ids.tolist() == ["run1", "run1", "run2", "run2"]
    assert len(label_to_state) == 2


def test_pace_exact_from_chain_states():
    chain_states = {
        "run1": np.array(
            [
                [1, 1, 2, 2],
                [1, 2, 1, 2],
            ]
        ),
        "run2": np.array(
            [
                [9, 9, 8, 8],
                [3, 4, 3, 4],
            ]
        ),
    }

    result = pace_exact_from_chain_states(
        chain_states,
        relabel=True,
        threshold=0.1,
        apply_correction=False,
    )

    assert result.delta == 0.0
    assert result.converged is True


def test_pace_requires_matching_lengths():
    with pytest.raises(PACEError):
        pace_from_labels([0, 1], ["a"])