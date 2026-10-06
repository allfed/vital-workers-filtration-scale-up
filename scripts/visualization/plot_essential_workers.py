#!/usr/bin/env python3
"""World maps of essential and vital worker shares (% of labour force).

Produces one manuscript figure: a 2x2 grid of total and indoor essential and
vital workers, sharing one colorbar.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from viz_common import (
    ESSENTIAL_WORKERS_RESULTS,
    apply_allfed_style,
    draw_world_choropleth,
    label_panel,
    save_figure,
)
from processing.paths import ESSENTIAL_WORKERS_VISUALIZATIONS

VMIN = 0.0
VMAX = 87.0
CMAP = "viridis"
LEGEND_LABEL = "% of labour force"

# Manuscript panels: (panel label, column, title)
PANELS = [
    ("a", "%Essential Workers", "Total essential workers (% of labour force)"),
    ("b", "%Indoor Essential Workers", "Indoor essential workers (% of labour force)"),
    ("c", "%Vital Workers", "Total vital workers (% of labour force)"),
    ("d", "%Indoor Vital Workers", "Indoor vital workers (% of labour force)"),
]


def plot_2x2_grid(df: pd.DataFrame, output_path: Path) -> None:
    """
    Four world maps of worker shares with one shared colorbar.

    Arguments:
        df (pandas.DataFrame): Country table with the four share columns.
        output_path (Path): PNG to write.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 6.9))
    fig.subplots_adjust(
        left=0.01, right=0.99, top=0.93, bottom=0.09, hspace=0.14, wspace=-0.05
    )

    mappable = None
    for ax, (letter, col, title) in zip(axes.ravel(), PANELS):
        mappable = draw_world_choropleth(
            ax,
            df.assign(pct=df[col] * 100.0),
            iso_col="Country Code",
            value_col="pct",
            cmap=CMAP,
            vmin=VMIN,
            vmax=VMAX,
        )
        label_panel(ax, letter)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=3)

    cax = fig.add_axes([0.30, 0.03, 0.40, 0.022])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label(LEGEND_LABEL, fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    save_figure(fig, output_path)


def main(output_dir: Path) -> None:
    """
    Draw the worker share maps.

    Arguments:
        output_dir (Path): Folder for the PNG.
    """
    apply_allfed_style()
    df = pd.read_csv(ESSENTIAL_WORKERS_RESULTS / "essential_workers_by_country.csv")
    plot_2x2_grid(
        df.dropna(subset=["Country Code"]), output_dir / "pct_workers_by_country_2x2.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ESSENTIAL_WORKERS_VISUALIZATIONS,
        help="Directory for PNG outputs",
    )
    args = parser.parse_args()
    main(args.output_dir)
