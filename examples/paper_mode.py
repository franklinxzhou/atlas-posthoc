"""Create a self-contained, two-panel manuscript figure.

Run from the repository root:

    python3 examples/paper_mode.py

The requested ``.png`` output becomes a vector ``.pdf`` while paper mode is
active. Existing plotting code can instead opt in process-wide by setting
``ATLAS_POSTHOC_PAPER_MODE=1`` before it starts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from atlas_posthoc.paper import paper_mode, paper_output_path
from atlas_posthoc.plotting import plot_municipal_split_trace_panel


def synthetic_traces() -> list[dict[str, object]]:
    """Return small deterministic traces that require no external data."""
    proposals = [0, 10_000, 20_000, 30_000, 40_000]
    return [
        {
            "chain_id": "Chain 1",
            "proposals": proposals,
            "n_cut_municipalities": [41, 34, 31, 29, 28],
            "ousted_population_share": [0.081, 0.069, 0.061, 0.058, 0.056],
            "geography_mode": "COUSUB",
        },
        {
            "chain_id": "Chain 2",
            "proposals": proposals,
            "n_cut_municipalities": [43, 36, 32, 30, 29],
            "ousted_population_share": [0.084, 0.072, 0.064, 0.060, 0.057],
            "geography_mode": "COUSUB",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/paper_mode_example.png"),
        help="Requested figure path; paper mode changes its suffix to .pdf.",
    )
    args = parser.parse_args()

    with paper_mode():
        figure, _ = plot_municipal_split_trace_panel(
            synthetic_traces(),
            title="Municipal split convergence",
            output_path=args.output,
        )
        destination = paper_output_path(args.output)

    plt.close(figure)
    print(f"Wrote manuscript figure: {destination}")


if __name__ == "__main__":
    main()
