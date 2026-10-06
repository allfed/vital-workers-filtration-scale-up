#!/usr/bin/env python3
"""Manuscript figures for the filtration scale-up (ALLFED style).

Reads the scenario results written by ``src/filtration_scale_up_model.py`` and
produces:
  - global supply scenarios for a COVID-like pathogen and a measles-like pathogen
  - global measles eCADR supply against the requirement at each mask efficiency
  - global coverage broken down by supply channel, PACs or CR boxes prioritized
  - regional eCADR supply and indoor vital coverage, at six months
  - regional indoor vital and essential coverage, at six months
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from viz_common import (
    ESSENTIAL_WORKERS_RESULTS,
    add_horizontal_colorbar,
    apply_allfed_style,
    draw_world_choropleth,
    expand_regions_to_countries,
    label_panel,
    save_figure,
)
import ecadr_requirements as ecadr  # noqa: E402
from processing.paths import (  # noqa: E402
    CR_BOXES_PRIORITIZED_RESULTS,
    ESSENTIAL_WORKERS_PARAMETERS,
    FILTRATION_SCALE_UP_VISUALIZATIONS,
    PACS_PRIORITIZED_RESULTS,
    read_parameters,
)

# cmasher colormap names; change these to try other palettes from the library.
# CMAP_RANGE drops the black tip of arctic_r so the highest values stay navy.
REGION_COVERAGE_CMAP = "arctic_r"
REGION_SUPPLY_CMAP = "arctic_r"
CMAP_RANGE = (0.0, 0.95)
REGION_COVERAGE_VMAX = 100.0
REGION_COVERAGE_ALPHA = 0.8
# Extra vertical gap between stacked map panels, as a fraction of panel height.
# Constrained layout's default is 0.02; raise this to pull the panels apart.
PANEL_HSPACE = 0.10
LEGEND_FRAMEON = False
# Panel-letter position on the stacked scenario panels (axes coordinates).
SCENARIO_COVERAGE_PANEL_LABEL_XY = (-0.08, 1.129)

# Supply channels, in stacking order, with manuscript labels
CHANNEL_LABELS = {
    "cr_box": "Corsi-Rosenthal boxes, new",
    "repurposed_cr_box": "Corsi-Rosenthal boxes, repurposed filters",
    "pac": "Portable air cleaners, new",
    "repurposed_pac": "Portable air cleaners, repurposed",
    "baghouse": "Coal baghouse bags, new",
    "repurposed_baghouse": "Coal baghouse bags, repurposed",
}

# ALLFED style-sheet colours, reordered for colourblind-safer stacking.
# Same technology = dark/light pair; technologies use blue / yellow / grey
# instead of the default cycle's adjacent greens.
STACK_CHANNEL_COLORS = {
    "cr_box": "#3D87CB",
    "repurposed_cr_box": "#85abda",
    "pac": "#F0B323",
    "repurposed_pac": "#f6cd85",
    "baghouse": "#6c7075",
    "repurposed_baghouse": "#d8d8d9",
}

SCENARIO_LABELS = {
    1: "Scenario 1: higher factory utilisation only",
    2: "Scenario 2: growth capped by meltblown supply",
    3: "Scenario 3: growth based on mask production during COVID-19",
}

SCENARIO_COLORS = {
    3: "#3A913F",
    2: "#3D87CB",
    1: "#F0B323",
}


def load_coverage(label: str, scenario: int, pathogen: str | None = None) -> pd.DataFrame:
    """
    Read one coverage table.

    Arguments:
        label (str): "vital" or "essential".
        scenario (int): 1, 2 or 3.
        pathogen (str or None): "covid" reads the COVID companion file.
            "measles" or None reads the settings-based (measles) file.

    Returns:
        pandas.DataFrame: Coverage by region and week, as percentages.
    """
    if pathogen == "covid":
        filename = f"coverage_{label}_covid_scenario{scenario}.csv"
    else:
        filename = f"coverage_{label}_scenario{scenario}.csv"
    path = PACS_PRIORITIZED_RESULTS / filename
    df = pd.read_csv(path)
    for column in ["coverage_median", "coverage_lower", "coverage_upper"]:
        df[column] *= 100.0
    return df


def load_regional_ecadr(scenario: int, week: int) -> pd.DataFrame:
    """
    Sum median weekly eCADR to each UN region.

    Arguments:
        scenario (int): 1, 2 or 3.
        week (int): Week to sum.

    Returns:
        pandas.DataFrame: Columns region and ecadr_l_per_s.
    """
    weekly = pd.read_csv(
        PACS_PRIORITIZED_RESULTS / f"weekly_ecadr_by_country_scenario{scenario}.csv",
        index_col=0,
    ).reset_index(names="Country Name")
    regions = pd.read_csv(ESSENTIAL_WORKERS_RESULTS / "essential_workers_by_country.csv")[
        ["Country Name", "Region"]
    ]
    merged = weekly.merge(regions, on="Country Name")
    column = str(week)
    if column not in merged.columns:
        raise ValueError(f"Week {week} not in weekly eCADR output")
    return (
        merged.groupby("Region", as_index=False)[column]
        .sum()
        .rename(columns={"Region": "region", column: "ecadr_l_per_s"})
    )


def load_requirements() -> pd.DataFrame:
    """
    Read the eCADR requirement of each region and of the world.

    Returns:
        pandas.DataFrame: Indexed by region, in L/s.
    """
    return pd.read_csv(
        PACS_PRIORITIZED_RESULTS / "requirements_by_region.csv", index_col="region"
    )


def when_label(week: int) -> str:
    """
    Plain-language time for a figure title.

    Arguments:
        week (int): Weeks since the start of the pandemic.

    Returns:
        str: "after 3 months" for week 13, "after 6 months" for week 26, and
            "at week N" otherwise.
    """
    if week % 13 == 0:
        return f"after {week // 13 * 3} months"
    return f"at week {week}"


def _draw_scenario_coverage_ax(
    ax, by_scenario, essential_level, last_week, title, show_xlabel
):
    """
    Draw global scenario coverage on one axes, up to full essential coverage.

    Arguments:
        ax (matplotlib.axes.Axes): Axes to draw on.
        by_scenario (dict): Global vital coverage by scenario.
        essential_level (float): Essential requirement as % of vital.
        last_week (int): Last week on the x-axis.
        title (str): Axes title.
        show_xlabel (bool): Whether to draw the x-axis label.
    """
    for scenario, label in SCENARIO_LABELS.items():
        df = by_scenario[scenario]
        (line,) = ax.plot(
            df.week,
            df.coverage_median,
            linewidth=2,
            linestyle="-",
            color=SCENARIO_COLORS[scenario],
            label=label,
        )
        ax.fill_between(
            df.week,
            df.coverage_lower,
            df.coverage_upper,
            color=line.get_color(),
            alpha=0.2,
            linewidth=0,
        )

    ax.axhspan(
        100.0,
        essential_level,
        color="dimgray",
        alpha=0.15,
        label=("Requirements range (indoor vital to indoor essential)"),
    )

    ax.set_xlim(1, last_week)
    ax.set_ylim(0, essential_level)
    if show_xlabel:
        ax.set_xlabel("Weeks since the start of the pandemic")
    ax.set_ylabel("% of indoor vital worker requirement")
    ax.grid(True, linestyle="--", alpha=0.4)
    right = ax.secondary_yaxis(
        "right",
        functions=(
            lambda y: y * 100.0 / essential_level,
            lambda y: y * essential_level / 100.0,
        ),
    )
    right.set_ylabel("% of indoor essential worker requirement")
    ax.set_title(title, fontweight="bold")


def mask_requirement_ratios() -> dict:
    """
    Global indoor vital requirement at each mask efficiency, relative to settings.

    The same mask efficiency is used in health care and elsewhere. Requirements
    are summed from the group table, which leaves out the few countries whose
    values are filled in from neighbours, so the ratios are close but not
    exact.

    Returns:
        dict: Mask efficiency to requirement ratio.
    """
    parameters, _ = read_parameters(ESSENTIAL_WORKERS_PARAMETERS)
    rooms = ecadr.load_room_types()
    qer = ecadr.qer_ratio(parameters)
    vital_workers = (
        pd.read_csv(ESSENTIAL_WORKERS_RESULTS / "essential_workers_by_group.csv")
        .groupby("occupational_group")["Indoor Vital Workers"]
        .sum()
    )

    def vital_requirement(mask_healthcare, mask_other):
        net = ecadr.scale_rooms(rooms, parameters, qer, mask_healthcare, mask_other)
        return (vital_workers * net.net.set_axis(rooms.occupational_group)).sum()

    base = vital_requirement(parameters["u_new_healthcare"], parameters["u_new_other"])
    return {
        mask: vital_requirement(mask, mask) / base
        for mask in ecadr.mask_efficiencies(parameters)
    }


def plot_measles_mask_requirements(output_path: Path, scenario: int = 2) -> None:
    """
    Global eCADR supply against the measles requirement at each mask efficiency.

    Supply is the global coverage times the requirement it was measured
    against, so it keeps the model's uncertainty interval. Each horizontal line
    is the full indoor vital requirement at one mask efficiency.

    Arguments:
        output_path (Path): PNG to write.
        scenario (int): 1, 2 or 3.
    """
    df = load_coverage("vital", scenario)
    df = df[df.region == "Global"]
    requirement = (
        load_requirements().loc["Global", "indoor_vital_ecadr_l_per_s"] / 1e9
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    (supply,) = ax.plot(
        df.week,
        df.coverage_median / 100 * requirement,
        linewidth=2,
        color=SCENARIO_COLORS[scenario],
    )
    ax.fill_between(
        df.week,
        df.coverage_lower / 100 * requirement,
        df.coverage_upper / 100 * requirement,
        color=supply.get_color(),
        alpha=0.2,
        linewidth=0,
    )
    ax.annotate(
        "Filtration supply",
        (df.week.iloc[-1], df.coverage_median.iloc[-1] / 100 * requirement),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        color=supply.get_color(),
        fontweight="bold",
        annotation_clip=False,
    )

    for mask, ratio in mask_requirement_ratios().items():
        ax.plot(
            [df.week.iloc[0], df.week.iloc[-1]],
            [requirement * ratio] * 2,
            linestyle="--",
            linewidth=1.5,
            color="0.35",
        )
        ax.annotate(
            f"Global requirement ({mask:.0%} efficiency masks)",
            (df.week.iloc[0], requirement * ratio),
            xytext=(4, 3),
            textcoords="offset points",
            va="bottom",
            color="0.35",
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1},
            zorder=5,
        )

    ax.set_xlim(1, df.week.max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Weeks since the start of the pandemic")
    ax.set_ylabel("Global eCADR (billion L/s)")
    ax.set_title(
        "Global filtration supply against indoor vital workforce requirements "
        "at different mask efficiencies\n"
        f"{SCENARIO_LABELS[scenario]}",
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_stacked_channels(
    output_path: Path,
    scenario: int,
    results_dir: Path = PACS_PRIORITIZED_RESULTS,
    title_suffix: str = "",
) -> None:
    """
    Global coverage over time, broken down by supply channel.

    Arguments:
        output_path (Path): PNG to write.
        scenario (int): 1, 2 or 3.
        results_dir (Path): Directory with ``ecadr_by_channel`` CSVs.
        title_suffix (str): Extra line for the figure title.
    """
    channels = pd.read_csv(
        results_dir / f"ecadr_by_channel_scenario{scenario}.csv", index_col="week"
    )
    requirement = pd.read_csv(
        results_dir / "requirements_by_region.csv", index_col="region"
    ).loc["Global", "indoor_vital_ecadr_l_per_s"]
    shares = 100.0 * channels[list(CHANNEL_LABELS)] / requirement

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(
        shares.index,
        *[shares[name] for name in CHANNEL_LABELS],
        labels=list(CHANNEL_LABELS.values()),
        colors=[STACK_CHANNEL_COLORS[name] for name in CHANNEL_LABELS],
        alpha=0.75,
    )
    ax.set_xlim(shares.index.min(), shares.index.max())
    ax.set_ylim(0, None)
    ax.set_xlabel("Weeks since the start of the pandemic")
    ax.set_ylabel("% of indoor vital worker requirement")
    title = (
        f"Global filtration supply by source: measles-level transmissibility\n"
        f"{SCENARIO_LABELS[scenario]}"
    )
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=LEGEND_FRAMEON,
        framealpha=0.9,
    )
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_supply_and_coverage_maps(output_path: Path, scenario: int, week: int) -> None:
    """
    Two-panel map: regional eCADR supply above, indoor vital coverage below.

    Arguments:
        output_path (Path): PNG to write.
        scenario (int): 1, 2 or 3.
        week (int): Week to map.
    """
    supply = load_regional_ecadr(scenario, week)
    supply = supply[supply.region != "Global"].copy()
    supply["ecadr_billion_l_per_s"] = supply.ecadr_l_per_s / 1e9
    supply_map = expand_regions_to_countries(supply, "region", "ecadr_billion_l_per_s")

    coverage = load_coverage("vital", scenario)
    at_week = coverage[(coverage.week == week) & (coverage.region != "Global")]
    if at_week.empty:
        raise ValueError(f"No vital coverage at week {week}")
    coverage_map = expand_regions_to_countries(at_week, "region", "coverage_median")

    fig, (ax_supply, ax_coverage) = plt.subplots(
        2, 1, figsize=(10, 9), layout="constrained"
    )
    fig.set_constrained_layout_pads(hspace=PANEL_HSPACE)

    supply_mappable = draw_world_choropleth(
        ax_supply,
        supply_map,
        iso_col="Country Code",
        value_col="ecadr_billion_l_per_s",
        cmap=REGION_SUPPLY_CMAP,
        cmap_range=CMAP_RANGE,
        vmin=0.0,
        alpha=REGION_COVERAGE_ALPHA,
    )
    label_panel(ax_supply, "a")
    ax_supply.set_title(
        f"Filtration supply by UN region {when_label(week)}",
        fontsize=12,
        fontweight="bold",
        pad=6,
    )

    coverage_mappable = draw_world_choropleth(
        ax_coverage,
        coverage_map,
        iso_col="Country Code",
        value_col="coverage_median",
        cmap=REGION_COVERAGE_CMAP,
        cmap_range=CMAP_RANGE,
        vmin=0.0,
        vmax=REGION_COVERAGE_VMAX,
        alpha=REGION_COVERAGE_ALPHA,
    )
    label_panel(ax_coverage, "b")
    ax_coverage.set_title(
        f"Indoor vital workers covered by filtration {when_label(week)}",
        fontsize=12,
        fontweight="bold",
        pad=6,
    )

    # The panels are on different scales, so each carries its own bar
    add_horizontal_colorbar(fig, supply_mappable, ax_supply, "eCADR (billion L/s)")
    add_horizontal_colorbar(
        fig, coverage_mappable, ax_coverage, "% of indoor vital workers covered"
    )

    save_figure(fig, output_path)


def plot_covid_measles_scenario_coverage(output_path: Path) -> None:
    """
    Supply scenarios for a COVID-like pathogen above and measles below.

    Each panel keeps the vital axis, the essential axis, and the grey band
    from full vital coverage to full essential coverage. The top of the axis
    is 100% of that pathogen's indoor essential requirement.

    Arguments:
        output_path (Path): PNG to write.
    """
    requirements = load_requirements()
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    for ax, pathogen, name, show_xlabel in (
        (ax_top, "covid", "COVID-19-level transmissibility", False),
        (ax_bottom, "measles", "Measles-level transmissibility", True),
    ):
        by_scenario = {}
        for scenario in SCENARIO_LABELS:
            df = load_coverage("vital", scenario, pathogen)
            by_scenario[scenario] = df[df.region == "Global"]
        interval = by_scenario[1].interval_percent.iloc[0]
        last_week = int(by_scenario[1].week.max())
        if pathogen == "covid":
            essential_col = "indoor_essential_ecadr_covid_l_per_s"
            vital_col = "indoor_vital_ecadr_covid_l_per_s"
        else:
            essential_col = "indoor_essential_ecadr_l_per_s"
            vital_col = "indoor_vital_ecadr_l_per_s"
        essential_level = 100.0 * (
            requirements.loc["Global", essential_col]
            / requirements.loc["Global", vital_col]
        )
        _draw_scenario_coverage_ax(
            ax,
            by_scenario,
            essential_level,
            last_week,
            title=(
                f"Filtration supply against workforce requirements\n"
                f"{name}"
            ),
            show_xlabel=show_xlabel,
        )
    label_x, label_y = SCENARIO_COVERAGE_PANEL_LABEL_XY
    label_panel(ax_top, "a", x=label_x, y=label_y)
    label_panel(ax_bottom, "b", x=label_x, y=label_y)
    fig.subplots_adjust(hspace=0.35, bottom=0.12)
    fig.legend(
        *ax_top.get_legend_handles_labels(),
        fontsize=9,
        loc="center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=LEGEND_FRAMEON,
        framealpha=0.9,
    )
    save_figure(fig, output_path)


def plot_covid_measles_coverage_maps(
    output_path: Path, scenario: int, week: int, workforce: str
) -> None:
    """
    Regional coverage for one workforce: COVID above, measles below.

    Arguments:
        output_path (Path): PNG to write.
        scenario (int): 1, 2 or 3.
        week (int): Week to map.
        workforce (str): "vital" or "essential".
    """
    panels = []
    for letter, pathogen, name in [
        ("a", "covid", "COVID-level transmissibility"),
        ("b", "measles", "Measles-level transmissibility"),
    ]:
        df = load_coverage(workforce, scenario, pathogen)
        at_week = df[(df.week == week) & (df.region != "Global")]
        if at_week.empty:
            raise ValueError(f"No {pathogen} {workforce} coverage at week {week}")
        panels.append(
            (
                letter,
                expand_regions_to_countries(at_week, "region", "coverage_median"),
                f"{name}: indoor {workforce} workers covered {when_label(week)}",
            )
        )

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), layout="constrained")
    fig.set_constrained_layout_pads(hspace=PANEL_HSPACE)
    mappable = None
    for ax, (letter, data, title) in zip(axes, panels):
        mappable = draw_world_choropleth(
            ax,
            data,
            iso_col="Country Code",
            value_col="coverage_median",
            cmap=REGION_COVERAGE_CMAP,
            cmap_range=CMAP_RANGE,
            vmin=0.0,
            vmax=REGION_COVERAGE_VMAX,
            alpha=REGION_COVERAGE_ALPHA,
        )
        label_panel(ax, letter)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
    add_horizontal_colorbar(
        fig, mappable, list(axes), f"% of indoor {workforce} worker requirement"
    )
    save_figure(fig, output_path)


def main(output_dir: Path, scenario: int) -> None:
    """
    Draw every filtration figure.

    Arguments:
        output_dir (Path): Folder for the PNGs.
        scenario (int): Scenario for the stacked figures, mask figure and maps.
    """
    apply_allfed_style()
    plot_covid_measles_scenario_coverage(output_dir / "scenario_coverage_covid_measles.png")
    plot_measles_mask_requirements(
        output_dir / "scenario_ecadr_measles_mask_efficiency.png", scenario
    )
    plot_stacked_channels(output_dir / "global_stacked_cadr.png", scenario)
    plot_stacked_channels(
        output_dir / "global_stacked_cadr_cr_boxes_prioritized.png",
        scenario,
        results_dir=CR_BOXES_PRIORITIZED_RESULTS,
        title_suffix="CR boxes prioritized (panel filters diverted from PACs)",
    )
    plot_supply_and_coverage_maps(
        output_dir / "supply_and_coverage_week26.png", scenario, 26
    )
    for week in [13, 26]:
        plot_covid_measles_coverage_maps(
            output_dir / f"vital_coverage_covid_measles_week{week}.png",
            scenario,
            week,
            workforce="vital",
        )
    plot_covid_measles_coverage_maps(
        output_dir / "essential_coverage_covid_measles_week26.png",
        scenario,
        26,
        workforce="essential",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FILTRATION_SCALE_UP_VISUALIZATIONS,
        help="Directory for PNG outputs",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Scenario for the stacked figures, mask figure and maps",
    )
    args = parser.parse_args()
    main(args.output_dir, args.scenario)
