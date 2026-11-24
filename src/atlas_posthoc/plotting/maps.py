from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt

"""
Map plotting and reference-plan assignment utilities.

This module includes a self-contained replacement for the old in-house
`hierarchy.assignSmallUnits` helper. The goal is not to change the assignment
logic, but to make it package-safe:

- no dependency on local research-directory modules such as `hierarchy.py`,
  `countyFunctions`, or `shapefileToGraph`;
- no import-time reloads or progress-bar side effects;
- explicit geometry, CRS, and column validation;
- copy-on-write behavior rather than mutating the caller's GeoDataFrame;
- clear package-specific exceptions when assignment fails.

The assignment rule follows the original hierarchy workflow:
for each small unit, use spatial-index candidates from the reference plan;
if there is one candidate, assign to it; if there are multiple candidates,
choose the one with largest overlap; if no valid overlap exists, fall back to
the nearest reference geometry.
"""

from atlas_posthoc.plotting.style import (
    PlotStyle,
    PlotStyleName,
    apply_axis_labels,
    get_plot_style,
    plot_style,
    save_figure,
)


class MapPlotError(Exception):
    """Base exception for map plotting helpers."""


def _load_gdf(path_or_gdf: str | Path | gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if isinstance(path_or_gdf, gpd.GeoDataFrame):
        return path_or_gdf.copy()

    return gpd.read_file(path_or_gdf)


def validate_geometry(gdf: gpd.GeoDataFrame, *, name: str) -> gpd.GeoDataFrame:
    if "geometry" not in gdf.columns:
        raise MapPlotError(f"{name} does not have a geometry column.")

    out = gdf[gdf.geometry.notna()].copy()

    if out.empty:
        raise MapPlotError(f"{name} has no non-missing geometries.")

    return out


def assign_small_units(
    small_gdf: gpd.GeoDataFrame,
    large_gdf: gpd.GeoDataFrame,
    large_name: str,
) -> gpd.GeoDataFrame:
    """
    Assign each small unit to a large/reference unit.

    This is the self-contained atlas-posthoc replacement for the old
    `hierarchy.assignSmallUnits` helper.

    It deliberately preserves the old assignment rule:

    1. Use the large-unit spatial index to find bounding-box candidates.
    2. If exactly one candidate exists, assign the small unit to that candidate.
    3. If multiple candidates exist, assign to the candidate with the largest
       intersection area relative to the small-unit area.
    4. If no candidate has positive overlap, assign to the nearest large
       geometry.

    Improvements over the old helper:

    - does not require importing the in-house `hierarchy.py` file;
    - does not depend on local modules such as `countyFunctions` or
      `shapefileToGraph`;
    - validates geometry and required columns before assignment;
    - aligns CRS when both GeoDataFrames have CRS metadata;
    - returns a copy instead of mutating the input GeoDataFrame in place;
    - raises `MapPlotError` instead of printing warnings or failing silently.

    Parameters
    ----------
    small_gdf:
        Small units, such as precincts.

    large_gdf:
        Large/reference units, such as enacted districts.

    large_name:
        Column in `large_gdf` whose value should be assigned to each small unit.

    Returns
    -------
    GeoDataFrame
        Copy of `small_gdf` with a new/updated `large_name` column.
    """
    if large_name not in large_gdf.columns:
        raise MapPlotError(
            f"large_name column {large_name!r} not found in large_gdf. "
            f"Available columns: {list(large_gdf.columns)}"
        )

    small = validate_geometry(small_gdf, name="small_gdf").copy()
    large = validate_geometry(large_gdf, name="large_gdf").copy()

    if small.crs is not None and large.crs is not None and small.crs != large.crs:
        large = large.to_crs(small.crs)

    spatial_index = large.sindex
    sm_index_to_lg: dict[Any, Any] = {}

    large_geoms = large.geometry

    for sm_ind in small.index:
        sm_geom = small.loc[sm_ind, "geometry"]

        if sm_geom is None or sm_geom.is_empty:
            raise MapPlotError(f"small geometry at index {sm_ind!r} is missing/empty")

        possible_matches = list(spatial_index.intersection(sm_geom.bounds))
        matched = False

        if len(possible_matches) == 1:
            loc_ind = possible_matches[0]
            lg_ind = large.index[loc_ind]
            sm_index_to_lg[sm_ind] = large.loc[lg_ind, large_name]
            matched = True

        elif len(possible_matches) > 1:
            candidates = large.iloc[possible_matches]
            intersections = candidates.geometry.intersection(sm_geom)

            if sm_geom.area > 0:
                overlap_scores = intersections.area / sm_geom.area
            else:
                overlap_scores = intersections.area

            lg_ind = overlap_scores.idxmax()
            max_overlap = overlap_scores.max()

            if max_overlap > 0:
                sm_index_to_lg[sm_ind] = large.loc[lg_ind, large_name]
                matched = True

        if not matched:
            distances = large_geoms.distance(sm_geom)
            lg_ind = distances.idxmin()
            sm_index_to_lg[sm_ind] = large.loc[lg_ind, large_name]

    small[large_name] = small.index.map(lambda idx: sm_index_to_lg[idx])
    return small


def assign_small_units_to_reference_plan(
    small_units: str | Path | gpd.GeoDataFrame,
    reference_plan: str | Path | gpd.GeoDataFrame,
    *,
    district_col: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Align CRS and assign small units to a reference/enacted district plan.

    This wraps `assign_small_units` with file loading, geometry validation,
    CRS alignment, and district-column checks. It is the package-safe version
    of the common notebook pattern:

        pctSF = gpd.read_file(...)
        enacted = gpd.read_file(...)
        enacted = enacted.to_crs(pctSF.crs)
        pctSF = hierarchy.assignSmallUnits(pctSF, enacted, district_col)

    Unlike the old notebook workflow, this function makes the assumptions
    explicit and raises `MapPlotError` when the inputs are not usable.
    """
    small_gdf = validate_geometry(_load_gdf(small_units), name="small_units")
    reference_gdf = validate_geometry(_load_gdf(reference_plan), name="reference_plan")

    if district_col not in reference_gdf.columns:
        raise MapPlotError(
            f"district column {district_col!r} not found in reference plan. "
            f"Available columns: {list(reference_gdf.columns)}"
        )

    if small_gdf.crs is None:
        raise MapPlotError("small_units CRS is missing.")

    if reference_gdf.crs is None:
        raise MapPlotError("reference_plan CRS is missing.")

    reference_gdf = reference_gdf.to_crs(small_gdf.crs)
    reference_gdf = reference_gdf.dropna(subset=[district_col]).copy()

    if reference_gdf.empty:
        raise MapPlotError(
            f"reference_plan has no non-missing values in {district_col!r}."
        )

    assigned = assign_small_units(
        small_gdf,
        reference_gdf,
        district_col,
    )

    if district_col not in assigned.columns:
        raise MapPlotError(
            f"After assignment, {district_col!r} was not found in the small-unit "
            "GeoDataFrame."
        )

    return assigned, reference_gdf


def plot_reference_plan_map(
    small_units: gpd.GeoDataFrame,
    *,
    district_col: str,
    reference_boundaries: gpd.GeoDataFrame | None = None,
    title: str | None = None,
    style: PlotStyleName | PlotStyle = "manuscript",
    figsize: tuple[float, float] | None = None,
    small_unit_edgecolor: str = "white",
    small_unit_linewidth: float = 0.08,
    boundary_edgecolor: str = "black",
    boundary_linewidth: float = 1.0,
    legend: bool = False,
    ax: Any | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """
    Plot a reference/enacted district plan assigned to small units.
    """
    if district_col not in small_units.columns:
        raise MapPlotError(
            f"district column {district_col!r} not found in small_units."
        )

    s = get_plot_style(style)

    if figsize is None:
        figsize = s.figsize_square

    if ax is None:
        with plot_style(s):
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    small_units.plot(
        column=district_col,
        ax=ax,
        categorical=True,
        legend=legend,
        edgecolor=small_unit_edgecolor,
        linewidth=small_unit_linewidth,
    )

    if reference_boundaries is not None:
        reference_boundaries.boundary.plot(
            ax=ax,
            edgecolor=boundary_edgecolor,
            linewidth=boundary_linewidth,
        )

    ax.grid(False, which="both")
    ax.set_axis_off()
    apply_axis_labels(ax, title=title)

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig, ax


def make_reference_plan_map(
    small_units: str | Path | gpd.GeoDataFrame,
    reference_plan: str | Path | gpd.GeoDataFrame,
    *,
    district_col: str,
    title: str | None = None,
    style: PlotStyleName | PlotStyle = "manuscript",
    figsize: tuple[float, float] | None = None,
    legend: bool = False,
    output_path: str | Path | None = None,
) -> tuple[Any, Any, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Assign small units to a reference plan and plot the result.
    """
    assigned, aligned_reference = assign_small_units_to_reference_plan(
        small_units,
        reference_plan,
        district_col=district_col,
    )

    fig, ax = plot_reference_plan_map(
        assigned,
        district_col=district_col,
        reference_boundaries=aligned_reference,
        title=title,
        style=style,
        figsize=figsize,
        legend=legend,
        output_path=output_path,
    )

    return fig, ax, assigned, aligned_reference


def make_oh_enacted_congressional_map(
    *,
    precincts_path: str | Path = "data/derived/pctSF_OH",
    enacted_path: str | Path = "data/raw/oh_cong_adopted_2022",
    district_col: str = "CONG_DIST",
    output_path: str | Path | None = "outputs/figures/OH_Enacted_Map.png",
    title: str | None = "Ohio enacted congressional plan",
    style: PlotStyleName | PlotStyle = "manuscript",
    figsize: tuple[float, float] = (10.0, 10.0),
) -> tuple[Any, Any, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Convenience wrapper for the Ohio enacted congressional map.
    """
    return make_reference_plan_map(
        precincts_path,
        enacted_path,
        district_col=district_col,
        title=title,
        style=style,
        figsize=figsize,
        legend=False,
        output_path=output_path,
    )