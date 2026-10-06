#!/usr/bin/env python3
"""Stacked composition of essential / vital workforces by occupational group.

Produces a 2×2 figure (essential, indoor essential, vital, indoor vital) of
the global worker-weighted shares in group_composition_global.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from viz_common import ESSENTIAL_WORKERS_RESULTS, apply_allfed_style, save_figure
from processing.paths import ESSENTIAL_WORKERS_VISUALIZATIONS

PANELS = [
    ("a", "% of Essential Workers", "Essential workforce"),
    ("b", "% of Indoor Essential Workers", "Indoor essential workforce"),
    ("c", "% of Vital Workers", "Vital workforce"),
    ("d", "% of Indoor Vital Workers", "Indoor vital workforce"),
]


def plot_global_composition(global_df: pd.DataFrame, output_path: Path) -> None:
    """
    Stacked bars of each workforce's make-up by occupational group.

    Arguments:
        global_df (pandas.DataFrame): group_composition_global.csv.
        output_path (Path): PNG to write.
    """
    groups = list(global_df["occupational_group"])
    # Distinct qualitative colors (avoid default cycle collisions).
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(groups))]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    fig.subplots_adjust(
        left=0.07, right=0.78, top=0.92, bottom=0.08, hspace=0.35, wspace=0.25
    )

    for ax, (letter, col, title) in zip(axes.ravel(), PANELS):
        shares = global_df[col].to_numpy(dtype=float) * 100.0
        left = 0.0
        for share, color in zip(shares, colors):
            ax.barh(0, share, left=left, height=0.55, color=color, edgecolor="none")
            left += share
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("% of workforce", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(
            0.01,
            1.12,
            f"({letter})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
        )
        # Percent labels for large segments.
        left = 0.0
        for share in shares:
            if share >= 8.0:
                ax.text(
                    left + share / 2.0,
                    0.0,
                    f"{share:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
            left += share

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=g) for g, c in zip(groups, colors)
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.80, 0.5),
        frameon=False,
        fontsize=9,
        title="Occupational group",
    )

    save_figure(fig, output_path)


def main(output_dir: Path) -> None:
    """
    Print and plot the global group composition.

    Arguments:
        output_dir (Path): Folder for the PNG.
    """
    apply_allfed_style()
    global_df = pd.read_csv(ESSENTIAL_WORKERS_RESULTS / "group_composition_global.csv")
    share_columns = [column for _, column, _ in PANELS]
    print("Global composition (% of category workforce):")
    print(
        (global_df.set_index("occupational_group")[share_columns] * 100)
        .round(1)
        .to_string()
    )
    plot_global_composition(global_df, output_dir / "group_composition_global.png")


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
