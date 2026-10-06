"""
Checks and side analyses of the essential-worker estimates.

  - The ILO validation: our essential share against the ILO's published share
    for each country, before and after the per-country overlap calibration.
  - The calibration detail: each country's overlap for each group.
  - The indoor sensitivity: global totals under each indoor-fraction method.
  - Range compression: whether indoor shares vary less between countries
    than total shares.
  - GDP: how worker shares relate to GDP per capita.

Run this file to write the tables to results/essential_workers/validation/.
"""

import numpy as np
import pandas as pd
from scipy import stats

import essential_workers as ew
from processing.paths import (
    ESSENTIAL_WORKERS_DATA,
    ESSENTIAL_WORKERS_PARAMETERS,
    ESSENTIAL_WORKERS_VALIDATION,
)
from processing.preprocessing import GROUPS, INDOOR_CONTEXT_METHODS

GDP_PPP_COL = "GDP per capita, PPP (current international $)"
GDP_USD_COL = "GDP per capita (current US$)"
GDP_FILE = ESSENTIAL_WORKERS_DATA / "gdp_per_capita_wdi.csv"
LF_SHARE_COLS = [
    "%Essential Workers",
    "%Indoor Essential Workers",
    "%Vital Workers",
    "%Indoor Vital Workers",
]
COMPOSITION_COUNTS = [
    "Essential Workers",
    "Indoor Essential Workers",
    "Vital Workers",
    "Indoor Vital Workers",
]
FOOD_SHARE_COLS = [f"Food % of {column}" for column in COMPOSITION_COUNTS]


def model_essential_pct(results):
    """
    Each country's essential share before calibration, using global overlaps.

    Arguments:
        results (dict): Output of essential_workers.estimate.

    Returns:
        pandas.Series: Percent of total employment, indexed by country name.
    """
    shares, _ = ew.worker_shares(
        results["employment"],
        results["weights_template"],
        {},
        ew.group_overlaps(results["parameters"]),
    )
    return 100.0 * ew.country_shares(shares).set_index("Country Name")[
        "%Essential Workers"
    ]


def calibrated_essential_pct(results):
    """
    Each country's calibrated essential share, for countries with ILO data.

    Arguments:
        results (dict): Output of essential_workers.estimate.

    Returns:
        pandas.Series: Percent of total employment, indexed by country name.
    """
    shares, _ = ew.worker_shares(
        results["employment"],
        results["weights_template"],
        results["overlaps"],
        ew.group_overlaps(results["parameters"]),
    )
    return 100.0 * ew.country_shares(shares).set_index("Country Name")[
        "%Essential Workers"
    ]


def validate_against_ilo(results):
    """
    Compare our essential share with the ILO's published share by country.

    The ILO figure is built from worker-level data, so the gap measures the
    cost of using one overlap per occupational group.

    Arguments:
        results (dict): Output of essential_workers.estimate.

    Returns:
        dict: table (one row per country, calibrated and model shares and
            their gaps to the ILO), mean_abs_delta_pp, correlation and
            mean_ilo_pct_essential for the calibrated shares, and outliers
            (countries further than ilo_outlier_threshold_pp from the ILO).
    """
    table = results["by_country"].merge(
        results["ilo_published"], on="Country Name", how="inner"
    )
    ilo = table["ILO %essential (published)"]
    table["Our %Essential (calibrated)"] = table["%Essential Workers"] * 100
    table["Our %Essential (model, global overlap)"] = table["Country Name"].map(
        model_essential_pct(results)
    )
    table["Delta model (pp)"] = table["Our %Essential (model, global overlap)"] - ilo
    table["Delta calibrated (pp)"] = table["Our %Essential (calibrated)"] - ilo
    table = table[
        [
            "Country Name",
            "Country Code",
            ew.LABOUR_FORCE_COL,
            "Essential Workers",
            "%Essential Workers",
            "Our %Essential (model, global overlap)",
            "Our %Essential (calibrated)",
            "ILO %essential (published)",
            "ILO %essential non-agri (published)",
            "Delta model (pp)",
            "Delta calibrated (pp)",
        ]
    ]
    gap = table["Delta calibrated (pp)"].abs()
    threshold = results["parameters"]["ilo_outlier_threshold_pp"]
    return {
        "table": table,
        "mean_abs_delta_pp": float(gap.mean()),
        "correlation": float(table["Our %Essential (calibrated)"].corr(ilo)),
        "mean_ilo_pct_essential": float(ilo.mean()),
        "outliers": table.loc[gap[gap > threshold].sort_values(ascending=False).index],
    }


def calibration_detail(results):
    """
    Each country's global and calibrated overlap for every group.

    Arguments:
        results (dict): Output of essential_workers.estimate.

    Returns:
        pandas.DataFrame: One row per country and group.
    """
    baseline = ew.group_overlaps(results["parameters"])
    ilo_lookup = (
        results["ilo_published"]
        .set_index("Country Name")["ILO %essential (published)"]
        .to_dict()
    )
    model_pct = model_essential_pct(results).to_dict()
    calibrated_pct = calibrated_essential_pct(results).to_dict()
    records = []
    for _, row in results["calibration"].iterrows():
        country = row["Country Name"]
        codes = results["employment"].get(country)
        masses = (
            ew.essential_mass_by_group(codes, results["weights_template"]) if codes else {}
        )
        ilo_pct = ilo_lookup.get(country, np.nan)
        for group in GROUPS:
            overlap = (
                baseline[group] if group == "ArmedForces" else row[f"overlap_{group}"]
            )
            records.append(
                {
                    "Country Name": country,
                    "Country Code": row["Country Code"],
                    "Group": group,
                    "Global overlap": baseline[group],
                    "Calibrated overlap": overlap,
                    "Adjustment": overlap - baseline[group],
                    "Group essential mass S_g": masses.get(group, np.nan),
                    "Overlap source": row["overlap_source"],
                    "calibration_x": row["calibration_x"],
                    "ILO %essential (published)": ilo_pct,
                    "Model %Essential (pct)": model_pct.get(country, np.nan),
                    "Calibrated %Essential (pct)": calibrated_pct.get(country, np.nan),
                    "Delta model (pp)": model_pct.get(country, np.nan) - ilo_pct,
                    "Delta calibrated (pp)": calibrated_pct.get(country, np.nan)
                    - ilo_pct,
                    "solver_status": row["solver_status"],
                }
            )
    return pd.DataFrame(records)


def global_worker_summary(by_country):
    """
    Global essential and vital workers, split into indoor and outdoor.

    Outdoor workers are the total minus the indoor workers.

    Arguments:
        by_country (pandas.DataFrame): Country table from estimate.

    Returns:
        pandas.DataFrame: Indexed by Category, with Workers, % of Labour Force
            and the lowest and highest country shares (percent). The global
            labour force is in .attrs["labour_force"].
    """
    rows, country_pct = {}, {}
    for name in ["Essential", "Vital"]:
        total = by_country[f"{name} Workers"].sum()
        indoor = by_country[f"Indoor {name} Workers"].sum()
        total_pct = by_country[f"%{name} Workers"] * 100.0
        indoor_pct = by_country[f"%Indoor {name} Workers"] * 100.0
        rows[f"{name} workers"] = total
        rows[f"Indoor {name.lower()} workers"] = indoor
        rows[f"Outdoor {name.lower()} workers"] = total - indoor
        country_pct[f"{name} workers"] = total_pct
        country_pct[f"Indoor {name.lower()} workers"] = indoor_pct
        country_pct[f"Outdoor {name.lower()} workers"] = total_pct - indoor_pct

    labour_force = by_country[ew.LABOUR_FORCE_COL].sum()
    summary = pd.DataFrame(
        {
            "Workers": rows,
            "% of Labour Force": {k: 100 * v / labour_force for k, v in rows.items()},
            "Country min %": {k: s.min() for k, s in country_pct.items()},
            "Country max %": {k: s.max() for k, s in country_pct.items()},
        }
    )
    summary.index.name = "Category"
    summary.attrs["labour_force"] = labour_force
    return summary


def indoor_context_sensitivity(
    parameters_file=ESSENTIAL_WORKERS_PARAMETERS,
    data_dir=ESSENTIAL_WORKERS_DATA,
    methods=INDOOR_CONTEXT_METHODS,
):
    """
    Global worker totals under each indoor-fraction method.

    Arguments:
        parameters_file (Path): Essential-worker parameters file.
        data_dir (Path): Folder holding the essential-worker data files.
        methods (tuple): Indoor-fraction methods to compare.

    Returns:
        pandas.DataFrame: The global_worker_summary rows for every method.
    """
    rows = []
    for method in methods:
        results = ew.estimate(parameters_file, data_dir, indoor_context_method=method)
        summary = global_worker_summary(results["by_country"]).reset_index()
        rows.append(summary.assign(indoor_context_method=method))
    return pd.concat(rows, ignore_index=True)


def pitman_morgan_variance_test(x, y):
    """
    Pitman-Morgan test of equal variances for paired observations.

    Equal variances are the same as Corr(X + Y, X - Y) = 0 (Pitman 1939;
    Morgan 1939).

    Arguments:
        x (array-like): First series.
        y (array-like): Paired second series.

    Returns:
        dict: n, sd_x, sd_y, variance_ratio_y_over_x, p_two_sided and
            p_y_smaller (one-sided, for Var(Y) < Var(X)).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 3:
        return dict.fromkeys(
            ["sd_x", "sd_y", "variance_ratio_y_over_x", "p_two_sided", "p_y_smaller"],
            float("nan"),
        ) | {"n": float(x.size)}
    sd_x, sd_y = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
    r, p_two = stats.pearsonr(x + y, x - y)
    return {
        "n": float(x.size),
        "sd_x": sd_x,
        "sd_y": sd_y,
        "variance_ratio_y_over_x": sd_y**2 / sd_x**2 if sd_x > 0 else float("nan"),
        "p_two_sided": float(p_two),
        "p_y_smaller": float(p_two / 2 if r > 0 else 1 - p_two / 2),
    }


def paired_relative_spread_stats(x, y):
    """
    Compare the relative spread of two paired positive series.

    Arguments:
        x (array-like): First series.
        y (array-like): Paired second series.

    Returns:
        dict: mean_x, mean_y, cv_x, cv_y (SD over mean), cv_ratio_y_over_x
            and p_log_y_smaller (Pitman-Morgan on the logs, one-sided).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[keep], y[keep]
    if x.size < 3:
        return dict.fromkeys(
            ["mean_x", "mean_y", "cv_x", "cv_y", "cv_ratio_y_over_x", "p_log_y_smaller"],
            float("nan"),
        )
    cv_x = np.std(x, ddof=1) / np.mean(x)
    cv_y = np.std(y, ddof=1) / np.mean(y)
    return {
        "mean_x": float(np.mean(x)),
        "mean_y": float(np.mean(y)),
        "cv_x": float(cv_x),
        "cv_y": float(cv_y),
        "cv_ratio_y_over_x": float(cv_y / cv_x),
        "p_log_y_smaller": pitman_morgan_variance_test(np.log(x), np.log(y))[
            "p_y_smaller"
        ],
    }


def indoor_range_compression(by_country):
    """
    Whether indoor shares vary less between countries than total shares.

    p-values are Benjamini-Hochberg adjusted: the four one-sided tests
    together, and the two two-sided tests together.

    Arguments:
        by_country (pandas.DataFrame): Country table from estimate.

    Returns:
        pandas.DataFrame: One row each for essential and vital workers.
    """
    rows = []
    for label, total_col, indoor_col in [
        ("Essential → Indoor essential", "%Essential Workers", "%Indoor Essential Workers"),
        ("Vital → Indoor vital", "%Vital Workers", "%Indoor Vital Workers"),
    ]:
        paired = by_country[[total_col, indoor_col]].dropna() * 100.0
        total, indoor = paired[total_col], paired[indoor_col]
        total_range = float(total.max() - total.min())
        indoor_range = float(indoor.max() - indoor.min())
        variance = pitman_morgan_variance_test(total, indoor)
        relative = paired_relative_spread_stats(total, indoor)
        rows.append(
            {
                "Transition": label,
                "n countries": int(variance["n"]),
                "Total country min %": float(total.min()),
                "Total country max %": float(total.max()),
                "Total range (pp)": total_range,
                "Total SD (pp)": variance["sd_x"],
                "Total mean %": relative["mean_x"],
                "Total CV": relative["cv_x"],
                "Indoor country min %": float(indoor.min()),
                "Indoor country max %": float(indoor.max()),
                "Indoor range (pp)": indoor_range,
                "Indoor SD (pp)": variance["sd_y"],
                "Indoor mean %": relative["mean_y"],
                "Indoor CV": relative["cv_y"],
                "Δ range (pp)": indoor_range - total_range,
                "% change in range": (
                    (indoor_range - total_range) / total_range * 100.0
                    if total_range
                    else float("nan")
                ),
                "Variance ratio (indoor/total)": variance["variance_ratio_y_over_x"],
                "CV ratio (indoor/total)": relative["cv_ratio_y_over_x"],
                "Pitman–Morgan p (two-sided)": variance["p_two_sided"],
                "p (indoor SD < total)": variance["p_y_smaller"],
                "p (indoor log-SD < total)": relative["p_log_y_smaller"],
            }
        )
    out = pd.DataFrame(rows).set_index("Transition")
    one_sided = ["p (indoor SD < total)", "p (indoor log-SD < total)"]
    adjusted = stats.false_discovery_control(out[one_sided].to_numpy().ravel())
    out[one_sided] = adjusted.reshape(out[one_sided].shape)
    out["Pitman–Morgan p (two-sided)"] = stats.false_discovery_control(
        out["Pitman–Morgan p (two-sided)"]
    )
    return out


def food_share_of_workforce(by_group):
    """
    Food workers as a share of each country's essential and vital workforces.

    Arguments:
        by_group (pandas.DataFrame): Group table from estimate.

    Returns:
        pandas.DataFrame: Country Code and FOOD_SHARE_COLS, as fractions.
    """
    totals = by_group.groupby("Country Code")[COMPOSITION_COUNTS].sum()
    food = by_group[by_group.occupational_group == "Food"].set_index("Country Code")
    shares = food[COMPOSITION_COUNTS] / totals.loc[food.index]
    shares.columns = FOOD_SHARE_COLS
    return shares.reset_index()


def worker_shares_vs_gdp(by_country, by_group, gdp_file=GDP_FILE):
    """
    Join worker shares to GDP per capita and test how they are associated.

    The main test is Spearman's rank correlation with GDP per capita (PPP).
    Pearson's correlation with log GDP is also given. Both sets of p-values
    are Benjamini-Hochberg adjusted across the shares.

    Arguments:
        by_country (pandas.DataFrame): Country table from estimate.
        by_group (pandas.DataFrame): Group table from estimate.
        gdp_file (Path): World Bank GDP per capita table.

    Returns:
        tuple: (table of shares and GDP by country, correlation summary).
    """
    gdp = pd.read_csv(gdp_file)[
        ["Country Code", GDP_PPP_COL, "GDP Year (PPP)", "GDP Year (USD)", GDP_USD_COL]
    ].drop_duplicates("Country Code")
    merged = (
        by_country[["Country Name", "Country Code", "Region", *LF_SHARE_COLS]]
        .merge(gdp, on="Country Code", how="left")
        .merge(food_share_of_workforce(by_group), on="Country Code", how="left")
    )

    rows = []
    for column in LF_SHARE_COLS + FOOD_SHARE_COLS:
        pair = merged[[column, GDP_PPP_COL]].dropna()
        pair = pair[pair[GDP_PPP_COL] > 0]
        rho, rho_p = stats.spearmanr(pair[GDP_PPP_COL], pair[column])
        r, r_p = stats.pearsonr(np.log(pair[GDP_PPP_COL]), pair[column])
        rows.append(
            {
                "Share": (
                    f"{column} (% of labour force)" if column.startswith("%") else column
                ),
                "column": column,
                "n": float(len(pair)),
                "Spearman ρ": float(rho),
                "Spearman p": float(rho_p),
                "Pearson r (log GDP)": float(r),
                "Pearson p (log GDP)": float(r_p),
            }
        )
    summary = pd.DataFrame(rows)
    for column in ["Spearman p", "Pearson p (log GDP)"]:
        summary[column] = stats.false_discovery_control(summary[column])
    return merged, summary


def main():
    """Write every validation table."""
    ESSENTIAL_WORKERS_VALIDATION.mkdir(parents=True, exist_ok=True)
    results = ew.estimate()
    validation = validate_against_ilo(results)
    gdp_table, gdp_correlations = worker_shares_vs_gdp(
        results["by_country"], results["by_group"]
    )
    tables = {
        "essential_workers_validation.csv": validation["table"],
        "group_overlap_calibration.csv": calibration_detail(results),
        "indoor_context_sensitivity.csv": indoor_context_sensitivity(),
        "worker_shares_vs_gdp.csv": gdp_table,
        "worker_shares_vs_gdp_correlations.csv": gdp_correlations,
    }
    for name, table in tables.items():
        table.to_csv(ESSENTIAL_WORKERS_VALIDATION / name, index=False)
    print(
        f"Calibrated essential share is {validation['mean_abs_delta_pp']:.2f} pp from "
        f"the ILO on average, {len(validation['outliers'])} countries over the "
        "outlier threshold."
    )
    print(f"Wrote {len(tables)} tables to {ESSENTIAL_WORKERS_VALIDATION}")


if __name__ == "__main__":
    main()
