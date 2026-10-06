"""
Fit the two linear models that feed the in-room filtration scale-up model.

Both were previously fitted outside the repository in a spreadsheet. They are
here so the whole calculation lives in one place:

  1. Coal plant capacity (MW) to baghouse airflow (L/s), from a small sample of
     plants. Used by methods equation 7.
  2. Filtration output against manufacturing value added. The scale-up samples
     the exponent b between the PRODCOM slope and the pooled slope.

Running this script writes one plot per model and writes the baghouse fit and
the two slopes of b into parameters_filtration_scale_up.csv.
"""

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from processing.paths import (
    FILTRATION_SCALE_UP_DATA,
    FILTRATION_SCALE_UP_PARAMETERS,
    LINEAR_MODELS_RESULTS,
    REPO_ROOT,
)

sys.path.insert(0, str(REPO_ROOT / "scripts" / "visualization"))
from viz_common import apply_allfed_style, label_panel  # noqa: E402

COAL_FILE = FILTRATION_SCALE_UP_DATA / "coal_plant_airflow.csv"
ALLOCATOR_FILE = FILTRATION_SCALE_UP_DATA / "allocator_fit_data.csv"

# Dataset labels used in the allocator input file
DATASETS = {
    "GrandView": "Filtration market revenue",
    "PRODCOM": "PRODCOM 28251410",
}


def fit_coal_airflow(path=COAL_FILE):
    """
    Fit baghouse airflow against coal plant capacity.

    Arguments:
        path (str or Path): CSV with columns plant, capacity_mw,
            airflow_l_per_s.

    Returns:
        dict: slope, intercept, r_squared and the sample size.
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(
            f"{path} is empty. Add the coal plant sample from the methods "
            "Supplementary Information (columns: plant, capacity_mw, airflow_l_per_s)."
        )

    model = sm.OLS(
        df.airflow_l_per_s.to_numpy(float),
        sm.add_constant(df.capacity_mw.to_numpy(float)),
    ).fit()
    params = np.asarray(model.params)
    return {
        "slope": params[1],
        "intercept": params[0],
        "r_squared": model.rsquared,
        "n": int(model.nobs),
        "data": df,
    }


def plot_coal_airflow(fit, path):
    """
    Plot the coal capacity to airflow fit.

    Arguments:
        fit (dict): Output of fit_coal_airflow.
        path (str): Where to save the figure.
    """
    df = fit["data"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(df.capacity_mw, df.airflow_l_per_s, zorder=3)
    x = np.linspace(0, df.capacity_mw.max() * 1.05, 50)
    ax.plot(x, fit["intercept"] + fit["slope"] * x, zorder=2)
    ax.set_xlabel("Coal plant capacity (MW)")
    ax.set_ylabel("Baghouse airflow (L/s)")
    ax.set_title(
        f"Airflow = {fit['slope']:.0f} × MW + {fit['intercept']:,.0f}\n"
        f"R² = {fit['r_squared']:.2f}, n = {fit['n']}",
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fit_allocator(path=ALLOCATOR_FILE):
    """
    Fit filtration output against MVA.

    The pooled model shares one slope across both datasets and gives each its
    own intercept, so the slope is not distorted by the difference in levels.
    Single-dataset fits are kept for the diagnostic plot and as the bounds of
    the sampled exponent b.

    Arguments:
        path (str or Path): CSV with columns country, dataset, value_usd,
            mva_usd.

    Returns:
        dict: Single-dataset fits, the pooled fit, and the input data.
    """
    df = pd.read_csv(path)
    df["log_y"] = np.log10(df.value_usd)
    df["log_x"] = np.log10(df.mva_usd)
    df["is_prodcom"] = (df.dataset == "PRODCOM").astype(float)

    single = {}
    for name in DATASETS:
        subset = df[df.dataset == name]
        single[name] = sm.OLS(
            subset.log_y.to_numpy(), sm.add_constant(subset.log_x.to_numpy())
        ).fit()

    pooled = sm.OLS(
        df.log_y.to_numpy(),
        sm.add_constant(np.column_stack([df.log_x, df.is_prodcom])),
    ).fit()
    return {"pooled": pooled, "single": single, "data": df}


def plot_allocator(fit, path):
    """
    Plot the PRODCOM fit, with market-revenue and pooled fits beside it.

    Arguments:
        fit (dict): Output of fit_allocator.
        path (str): Where to save the figure.
    """
    df = fit["data"]
    fig = plt.figure(figsize=(12.5, 5.5))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.9, 1], hspace=0.6, wspace=0.25)
    main = fig.add_subplot(grid[:, 0])
    market_ax = fig.add_subplot(grid[0, 1])
    pooled_ax = fig.add_subplot(grid[1, 1])

    # Main panel: PRODCOM only
    prodcom = df[df.dataset == "PRODCOM"]
    model = fit["single"]["PRODCOM"]
    intercept, slope = np.asarray(model.params)
    x = np.linspace(prodcom.log_x.min() - 0.12, prodcom.log_x.max() + 0.12, 50)
    main.scatter(prodcom.log_x, prodcom.log_y, zorder=3)
    main.plot(x, intercept + slope * x, zorder=2)
    main.annotate(
        f"b = {slope:.3f} ± {np.asarray(model.bse)[1]:.3f}\n"
        f"R² = {model.rsquared:.3f}",
        xy=(0.97, 0.04),
        xycoords="axes fraction",
        va="bottom",
        ha="right",
    )
    main.set_title(f"PRODCOM fit (n = {int(model.nobs)})", loc="left")
    main.set_xlabel("log$_{10}$ manufacturing value added (USD)")
    main.set_ylabel("log$_{10}$ annual value (USD)")
    label_panel(main, "a", x=-0.1, y=1.054)

    # Top side: filtration market revenue alone
    market = df[df.dataset == "GrandView"]
    model = fit["single"]["GrandView"]
    coeffs = np.asarray(model.params)
    xi = np.linspace(market.log_x.min() - 0.1, market.log_x.max() + 0.1, 50)
    market_ax.scatter(market.log_x, market.log_y, zorder=3)
    market_ax.plot(xi, coeffs[0] + coeffs[1] * xi, zorder=2)
    market_ax.annotate(
        f"b = {coeffs[1]:.3f} ± {np.asarray(model.bse)[1]:.3f}\n"
        f"R² = {model.rsquared:.3f}",
        xy=(0.96, 0.05),
        xycoords="axes fraction",
        va="bottom",
        ha="right",
        fontsize=8,
    )
    market_ax.set_title(
        f"Filtration market revenue fit (n = {int(model.nobs)})",
        loc="left",
        fontsize=9,
    )
    market_ax.set_ylabel("log$_{10}$ value", fontsize=9)
    label_panel(market_ax, "b", x=-0.27, y=1.14)

    # Bottom side: pooled fit across both datasets
    pooled = fit["pooled"]
    intercept, slope, shift = np.asarray(pooled.params)
    xi = np.linspace(df.log_x.min() - 0.1, df.log_x.max() + 0.1, 50)
    for name, label in DATASETS.items():
        subset = df[df.dataset == name]
        pooled_ax.scatter(subset.log_x, subset.log_y, label=label, zorder=3)
        kwargs = {"zorder": 2}
        if name == "PRODCOM":
            kwargs["color"] = "#3D87CB"
        pooled_ax.plot(
            xi,
            intercept + shift * (name == "PRODCOM") + slope * xi,
            **kwargs,
        )
    pooled_ax.annotate(
        f"b = {slope:.3f} ± {np.asarray(pooled.bse)[1]:.3f}\n"
        f"R² = {pooled.rsquared:.3f}",
        xy=(0.96, 0.05),
        xycoords="axes fraction",
        va="bottom",
        ha="right",
        fontsize=8,
    )
    n_prodcom = int((df.dataset == "PRODCOM").sum())
    n_market = int((df.dataset == "GrandView").sum())
    pooled_ax.set_title(
        f"Pooled fit (n = {n_prodcom} + {n_market})",
        loc="left",
        fontsize=9,
    )
    pooled_ax.set_xlabel("log$_{10}$ manufacturing value added (USD)", fontsize=9)
    pooled_ax.set_ylabel("log$_{10}$ value", fontsize=9)
    pooled_ax.legend(loc="upper left", fontsize=7)
    label_panel(pooled_ax, "c", x=-0.27, y=1.14)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fitted_rows(coal, allocator):
    """
    Parameter rows for the baghouse fit and the bounds of the MVA exponent b.

    Arguments:
        coal (dict): Output of fit_coal_airflow.
        allocator (dict): Output of fit_allocator.

    Returns:
        list: One dict per parameter row, keyed by column name.
    """
    source = str(COAL_FILE.relative_to(REPO_ROOT))
    prodcom_model = allocator["single"]["PRODCOM"]
    pooled_model = allocator["pooled"]
    prodcom = round(float(np.asarray(prodcom_model.params)[1]), 3)
    pooled = round(float(np.asarray(pooled_model.params)[1]), 3)
    low, high = sorted([prodcom, pooled])
    return [
        {
            "name": "baghouse_gradient",
            "value": round(coal["slope"], 2),
            "units": "L/s per MW",
            "source": source,
            "notes": "Fitted in linear_models.py. "
            f"R2 = {coal['r_squared']:.2f}, n = {coal['n']}.",
        },
        {
            "name": "baghouse_intercept_l_per_s",
            "value": round(coal["intercept"], 1),
            "units": "L/s",
            "source": source,
            "notes": "Fitted in linear_models.py. Per-plant intercept, so it is only "
            "applied to countries with non-zero coal capacity.",
        },
        {
            "name": "mva_exponent_b",
            "low": low,
            "high": high,
            "distribution": "normal",
            "units": "exponent",
            "source": str(ALLOCATOR_FILE.relative_to(REPO_ROOT)),
            "notes": (
                "90% interval bounds are the pooled slope "
                f"({pooled}, n = {int(pooled_model.nobs)}) and the PRODCOM slope "
                f"({prodcom}, n = {int(prodcom_model.nobs)})."
            ),
        },
    ]


def write_rows(rows, path=FILTRATION_SCALE_UP_PARAMETERS):
    """
    Write rows into a parameters file, replacing rows of the same name.

    Rows not already in the file are added at the end. Other rows keep their
    order and values.

    Arguments:
        rows (list): Dicts keyed by column name, each with a name.
        path (str or Path): Parameters CSV.
    """
    table = pd.read_csv(path).set_index("name").astype({"value": object})
    for row in rows:
        for column, value in row.items():
            if column != "name":
                table.loc[row["name"], column] = value
    table.reset_index().to_csv(path, index=False)


def main():
    """Fit both models, plot them and write the fits into the parameters file."""
    apply_allfed_style()
    print("=" * 70)
    print("LINEAR MODELS FEEDING THE SCALE-UP MODEL")
    print("=" * 70)
    LINEAR_MODELS_RESULTS.mkdir(parents=True, exist_ok=True)

    print("\nFitting coal capacity to baghouse airflow...")
    coal = fit_coal_airflow()
    print(
        f"  airflow = {coal['slope']:.0f} x MW + {coal['intercept']:,.0f}, "
        f"R2 = {coal['r_squared']:.2f}, n = {coal['n']}"
    )
    plot_coal_airflow(coal, LINEAR_MODELS_RESULTS / "coal_airflow_fit.png")

    print("\nFitting filtration output against MVA...")
    allocator = fit_allocator()
    pooled = allocator["pooled"]
    print(
        f"  pooled b = {np.asarray(pooled.params)[1]:.3f} "
        f"(SE {np.asarray(pooled.bse)[1]:.3f}), R2 = {pooled.rsquared:.3f}, "
        f"n = {int(pooled.nobs)}"
    )
    for name, model in allocator["single"].items():
        print(
            f"  {name:10s} b = {np.asarray(model.params)[1]:.3f}, "
            f"R2 = {model.rsquared:.3f}, n = {int(model.nobs)}"
        )
    plot_allocator(allocator, LINEAR_MODELS_RESULTS / "mva_allocator_fit.png")

    rows = fitted_rows(coal, allocator)
    write_rows(rows)
    print(f"\nPlots written to {LINEAR_MODELS_RESULTS}")
    print(f"Fitted values written into {FILTRATION_SCALE_UP_PARAMETERS}:")
    for row in rows:
        print(f"  {row['name']} = {row.get('value', (row.get('low'), row.get('high')))}")


if __name__ == "__main__":
    main()
