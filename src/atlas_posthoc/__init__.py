"""Post-hoc diagnostics and plotting for AtlasIO.jl redistricting chains."""

from atlas_posthoc.paper import (
    PANEL_LABEL_GID,
    PAPER_MODE_ENV,
    apply_paper_style,
    paper_mode,
    paper_mode_enabled,
    paper_output_path,
)


apply_paper_style()


__all__ = [
    "PANEL_LABEL_GID",
    "PAPER_MODE_ENV",
    "apply_paper_style",
    "paper_mode",
    "paper_mode_enabled",
    "paper_output_path",
]
