"""Repository paths, and the reader for the two parameter files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

ESSENTIAL_WORKERS_DATA = DATA_DIR / "essential_workers"
ESSENTIAL_WORKERS_PARAMETERS = ESSENTIAL_WORKERS_DATA / "parameters_essential_workers.csv"
FILTRATION_SCALE_UP_DATA = DATA_DIR / "filtration_scale_up"
FILTRATION_SCALE_UP_PARAMETERS = (
    FILTRATION_SCALE_UP_DATA / "parameters_filtration_scale_up.csv"
)

ESSENTIAL_WORKERS_RESULTS = RESULTS_DIR / "essential_workers"
ESSENTIAL_WORKERS_VALIDATION = ESSENTIAL_WORKERS_RESULTS / "validation"
ESSENTIAL_WORKERS_VISUALIZATIONS = ESSENTIAL_WORKERS_RESULTS / "visualizations"
FILTRATION_SCALE_UP_RESULTS = RESULTS_DIR / "filtration_scale_up"
PACS_PRIORITIZED_RESULTS = FILTRATION_SCALE_UP_RESULTS / "pacs_prioritized"
CR_BOXES_PRIORITIZED_RESULTS = FILTRATION_SCALE_UP_RESULTS / "cr_boxes_prioritized"
LINEAR_MODELS_RESULTS = FILTRATION_SCALE_UP_RESULTS / "linear_models"
FILTRATION_SCALE_UP_VISUALIZATIONS = FILTRATION_SCALE_UP_RESULTS / "visualizations"


def read_parameters(path):
    """
    Read a parameters file into fixed settings and Monte Carlo parameters.

    A row with a distribution is a Monte Carlo parameter, sampled between its
    low and high values. Any other row is a fixed setting held in value.

    Arguments:
        path (str or Path): Parameters CSV with columns name, value, low,
            high, distribution, units, source and notes.

    Returns:
        tuple: (dict of fixed settings, numeric where possible, and a
            DataFrame of Monte Carlo parameters indexed by name).
    """
    table = pd.read_csv(path).set_index("name")
    uncertain = table[table.distribution.notna()]
    missing = uncertain[(uncertain.low == 0) & (uncertain.high == 0)].index.tolist()
    if missing:
        raise ValueError(
            "These parameters have no value in the methods and must be set in "
            f"{path} before the model can run:\n  " + "\n  ".join(missing)
        )

    fixed = {}
    for name, value in table.loc[table.distribution.isna(), "value"].items():
        lowered = str(value).strip().lower()
        if lowered in ("true", "false"):
            fixed[name] = lowered == "true"
            continue
        try:
            fixed[name] = float(value)
        except ValueError:
            fixed[name] = value
    return fixed, uncertain
