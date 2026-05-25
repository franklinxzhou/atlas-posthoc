import matplotlib.pyplot as plt
import pytest
import geopandas as gpd
from shapely.geometry import box

from atlas_posthoc.plotting.maps import (
    MapPlotError,
    assign_small_units,
    assign_small_units_to_reference_plan,
    make_reference_plan_map,
    plot_reference_plan_map,
)


def make_large_gdf(crs="EPSG:3857"):
    return gpd.GeoDataFrame(
        {
            "DIST": [1, 2],
            "geometry": [
                box(0, 0, 1, 1),
                box(2, 0, 3, 1),
            ],
        },
        crs=crs,
    )


def make_small_gdf(crs="EPSG:3857"):
    return gpd.GeoDataFrame(
        {
            "id": [10, 11],
            "geometry": [
                box(0.1, 0.1, 0.9, 0.9),
                box(2.1, 0.1, 2.9, 0.9),
            ],
        },
        crs=crs,
    )


def test_assign_small_units_exact_single_candidate():
    small = make_small_gdf()
    large = make_large_gdf()

    assigned = assign_small_units(small, large, "DIST")

    assert assigned["DIST"].to_list() == [1, 2]

    # Should not mutate caller's input GeoDataFrame.
    assert "DIST" not in small.columns


def test_assign_small_units_largest_overlap():
    small = gpd.GeoDataFrame(
        {
            "id": [10],
            # Overlaps district 1 more than district 2.
            "geometry": [box(0.6, 0.1, 1.1, 0.9)],
        },
        crs="EPSG:3857",
    )

    large = gpd.GeoDataFrame(
        {
            "DIST": [1, 2],
            "geometry": [
                box(0, 0, 1, 1),
                box(1, 0, 2, 1),
            ],
        },
        crs="EPSG:3857",
    )

    assigned = assign_small_units(small, large, "DIST")

    assert assigned.loc[0, "DIST"] == 1


def test_assign_small_units_nearest_fallback_when_no_overlap():
    small = gpd.GeoDataFrame(
        {
            "id": [10],
            # No overlap with either large unit; closer to district 2.
            "geometry": [box(4.0, 0.0, 4.5, 0.5)],
        },
        crs="EPSG:3857",
    )

    large = make_large_gdf()

    assigned = assign_small_units(small, large, "DIST")

    assert assigned.loc[0, "DIST"] == 2


def test_assign_small_units_rejects_missing_large_column():
    small = make_small_gdf()
    large = make_large_gdf()

    with pytest.raises(MapPlotError):
        assign_small_units(small, large, "NOT_A_DISTRICT_COLUMN")


def test_assign_small_units_to_reference_plan_aligns_crs():
    small = make_small_gdf(crs="EPSG:3857")

    # Same geometries, but stored in a different CRS.
    reference = make_large_gdf(crs="EPSG:3857").to_crs("EPSG:4326")

    assigned, aligned_reference = assign_small_units_to_reference_plan(
        small,
        reference,
        district_col="DIST",
    )

    assert assigned["DIST"].to_list() == [1, 2]
    assert aligned_reference.crs == small.crs


def test_assign_small_units_to_reference_plan_rejects_missing_crs():
    small = make_small_gdf()
    reference = make_large_gdf()
    reference = reference.set_crs(None, allow_override=True)

    with pytest.raises(MapPlotError):
        assign_small_units_to_reference_plan(
            small,
            reference,
            district_col="DIST",
        )


def test_plot_reference_plan_map_writes_file(tmp_path):
    small = make_small_gdf()
    large = make_large_gdf()
    assigned = assign_small_units(small, large, "DIST")

    out = tmp_path / "reference_map.png"

    fig, ax = plot_reference_plan_map(
        assigned,
        district_col="DIST",
        reference_boundaries=large,
        title="Synthetic enacted map",
        output_path=out,
    )

    assert out.exists()
    assert not ax.axison

    plt.close(fig)


def test_make_reference_plan_map_writes_file(tmp_path):
    small = make_small_gdf()
    large = make_large_gdf()

    out = tmp_path / "made_reference_map.png"

    fig, ax, assigned, aligned_reference = make_reference_plan_map(
        small,
        large,
        district_col="DIST",
        title="Synthetic reference plan",
        output_path=out,
    )

    assert out.exists()
    assert assigned["DIST"].to_list() == [1, 2]
    assert aligned_reference.crs == small.crs

    plt.close(fig)


def test_plot_reference_plan_map_rejects_missing_district_col():
    small = make_small_gdf()

    with pytest.raises(MapPlotError):
        plot_reference_plan_map(
            small,
            district_col="DIST",
        )