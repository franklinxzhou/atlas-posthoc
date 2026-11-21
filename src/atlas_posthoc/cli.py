"""Command-line entry point for configuration-driven AtlasIO analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from atlas_posthoc.config import ConfigError, ProjectConfig, load_config
from atlas_posthoc.template import EXAMPLE_CONFIG

DEFAULT_CONFIG = "atlas-posthoc.toml"
WORKFLOW = "convergence"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas-posthoc",
        description="Run AtlasIO post-processing workflows from one TOML file.",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=DEFAULT_CONFIG,
        help=f"Project TOML configuration (default: {DEFAULT_CONFIG}).",
    )
    figure_mode = parser.add_mutually_exclusive_group()
    figure_mode.add_argument(
        "--paper-mode",
        dest="paper_mode",
        action="store_true",
        default=None,
        help="Write title-free, publication-style vector PDF figures.",
    )
    figure_mode.add_argument(
        "--no-paper-mode",
        dest="paper_mode",
        action="store_false",
        help="Override figures.paper_mode and write standard PNG figures.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List available workflows.")

    init = subparsers.add_parser(
        "init", help="Write a documented example configuration."
    )
    init.add_argument("path", nargs="?", default=DEFAULT_CONFIG)
    init.add_argument("--force", action="store_true")

    doctor = subparsers.add_parser(
        "doctor", help="Validate configuration and input paths."
    )
    doctor.add_argument("workflow", nargs="?", choices=[WORKFLOW], default=WORKFLOW)

    run = subparsers.add_parser("run", help="Run one workflow (default: convergence).")
    run.add_argument("workflow", nargs="?", choices=[WORKFLOW], default=WORKFLOW)
    run.add_argument(
        "--dry-run", action="store_true", help="Print resolved inputs and outputs."
    )
    return parser


def _init(path: str, *, force: bool) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("w" if force else "x", encoding="utf-8") as stream:
            stream.write(EXAMPLE_CONFIG)
    except FileExistsError as exc:
        raise ConfigError(
            f"refusing to overwrite existing file: {destination}"
        ) from exc
    print(destination)


def _show_run(config: ProjectConfig, *, paper: bool) -> None:
    print(f"[{WORKFLOW}]")
    print(f"Config: {config.path}")
    print(f"Graph: {config.graph_json}")
    print("Chains:")
    for path in config.chain_files:
        print(f"  {path}")
    print(f"Outputs: {config.output_root}")
    print(f"Figure format: {'PDF' if paper else 'PNG'}")


def _run(config: ProjectConfig, *, paper: bool) -> int:
    import matplotlib

    matplotlib.use("Agg")

    from atlas_posthoc.assignments.keys import AssignmentKeyError
    from atlas_posthoc.assignments.multiscale import MultiscaleAssignmentError
    from atlas_posthoc.diagnostics.rhat import RHatError
    from atlas_posthoc.diagnostics.total_variation import TotalVariationError
    from atlas_posthoc.io.atlas_reader import AtlasReaderError
    from atlas_posthoc.io.graph_reader import GraphReaderError
    from atlas_posthoc.observables.vote_share import VoteShareError
    from atlas_posthoc.pipeline.chain_observables import ChainObservableError
    from atlas_posthoc.pipeline.convergence import run_convergence

    try:
        report = run_convergence(config, paper=paper)
    except ConfigError:
        raise
    except (
        AssignmentKeyError,
        AtlasReaderError,
        ChainObservableError,
        GraphReaderError,
        MultiscaleAssignmentError,
        RHatError,
        TotalVariationError,
        VoteShareError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "list":
            print(f"{WORKFLOW:<18} Vote-share and compactness convergence diagnostics")
            return 0
        if args.command == "init":
            _init(args.path, force=args.force)
            return 0

        config = load_config(args.config)
        if args.command == "doctor":
            config.check_inputs()
            print(f"[{args.workflow}] Configuration and input paths are valid.")
            return 0

        paper = config.paper_mode if args.paper_mode is None else args.paper_mode
        if args.dry_run:
            _show_run(config, paper=paper)
            return 0
        return _run(config, paper=paper)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
