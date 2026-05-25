from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt

import hierarchy

"""
How to use this script: Call the following example command from the project root:
python3 scripts/make_oh_enacted_map.py \
  --precincts-path data/derived/pctSF_OH \
  --enacted-path data/raw/oh_cong_adopted_2022 \
  --district-col CONG_DIST \
  --output-path outputs/figures/OH_Enacted_Map.png
"""


def make_oh_enacted_map(
    precincts_path: str | Path = "data/derived/pctSF_OH",
    enacted_path: str | Path = "data/raw/oh_cong_adopted_2022",
    district_col: str = "CONG_DIST",
    output_path: str | Path = "outputs/figures/OH_Enacted_Map.png",
    figsize: tuple[float, float] = (10, 10),
    dpi: int = 300,
) -> Path:
    precincts_path = Path(precincts_path)
    enacted_path = Path(enacted_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pct_sf = gpd.read_file(precincts_path)
    enacted = gpd.read_file(enacted_path)

    if district_col not in enacted.columns:
        raise ValueError(
            f"District column {district_col!r} not found in enacted file. "
            f"Available columns: {list(enacted.columns)}"
        )

    pct_sf = pct_sf[pct_sf.geometry.notna()].copy()
    enacted = enacted[enacted.geometry.notna()].copy()

    enacted = enacted.to_crs(pct_sf.crs)
    enacted = enacted.dropna(subset=[district_col]).copy()

    # Assign each small unit / precinct to the enacted congressional district.
    pct_sf = hierarchy.assignSmallUnits(pct_sf, enacted, district_col)

    if district_col not in pct_sf.columns:
        raise ValueError(
            f"After hierarchy.assignSmallUnits, {district_col!r} was not found "
            "in the precinct GeoDataFrame."
        )

    fig, ax = plt.subplots(figsize=figsize)

    pct_sf.plot(
        column=district_col,
        ax=ax,
        edgecolor="white",
        linewidth=0.08,
        categorical=True,
        legend=False,
    )

    enacted.boundary.plot(
        ax=ax,
        edgecolor="black",
        linewidth=1.0,
    )

    ax.set_axis_off()
    ax.set_title("Ohio enacted congressional plan", fontsize=14)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the Ohio enacted congressional plan."
    )

    parser.add_argument(
        "--precincts-path",
        default="data/derived/pctSF_OH",
        help="Path to Ohio precinct/small-unit GeoDataFrame.",
    )
    parser.add_argument(
        "--enacted-path",
        default="data/raw/oh_cong_adopted_2022",
        help="Path to enacted Ohio congressional district shapefile/directory.",
    )
    parser.add_argument(
        "--district-col",
        default="CONG_DIST",
        help="District column in the enacted plan file.",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/figures/OH_Enacted_Map.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output image DPI.",
    )

    args = parser.parse_args()

    out = make_oh_enacted_map(
        precincts_path=args.precincts_path,
        enacted_path=args.enacted_path,
        district_col=args.district_col,
        output_path=args.output_path,
        dpi=args.dpi,
    )

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()