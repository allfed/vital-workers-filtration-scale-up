"""
Essential and vital workers in every country, in total and indoors.

Essential workers follow the ILO (2023) definition: a key occupation (ISCO-08)
in a key industry. Without worker-level data, the share of each occupational
group that works in a key industry (its overlap) starts from ILO Figure A1 and
is calibrated in each country to the ILO's published essential share. Vital
workers are the minimum subset of essential workers needed to maintain
basic services such as food, water, electricity, and healthcare. Indoor
workers weight each occupation by the share of its work done indoors.

Countries without ILO occupation data take the mean of similar countries
(data/essential_workers/similar_countries.csv). Each occupational group is
then given the filtration it needs, from ecadr_requirements.py.

Run this file to write the core results to results/essential_workers/. The
method is described in full in the README.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import ecadr_requirements as ecadr
from processing.paths import (
    ESSENTIAL_WORKERS_DATA,
    ESSENTIAL_WORKERS_PARAMETERS,
    ESSENTIAL_WORKERS_RESULTS,
    read_parameters,
)
from processing.preprocessing import GROUPS, read_inputs

LABOUR_FORCE_COL = "Labour Force (2024)"
WORKER_COLUMNS = [
    "Indoor Essential Workers",
    "Indoor Vital Workers",
    "Essential Workers",
    "Vital Workers",
]
PCT_COLUMNS = [f"%{column}" for column in WORKER_COLUMNS]
# ISCO weight column behind each worker count
WEIGHT_COLUMNS = {
    "Indoor Essential Workers": "ISCO_08_ILOWeights",
    "Indoor Vital Workers": "ISCO_08_PollWeights",
    "Essential Workers": "ISCO_08_ILOWeights_Total",
    "Vital Workers": "ISCO_08_PollWeights_Total",
}
SCALED_ECA_ESSENTIAL_COL = "Scaled ECA Essential (L/s/person)"
SCALED_ECA_VITAL_COL = "Scaled ECA Vital (L/s/person)"

CALIBRATABLE_GROUPS = [group for group in GROUPS if group != "ArmedForces"]
OVERLAP_COLUMNS = [f"overlap_{group}" for group in CALIBRATABLE_GROUPS]
OVERLAP_SOURCE_ILO = "ilo_calibrated"
OVERLAP_SOURCE_NEIGHBOUR = "neighbour_backfill"
OVERLAP_SOURCE_GLOBAL = "global_fallback"

# ISCO L2 codes excluded from on-site housing requirements: market-oriented
# skilled agricultural workers (61) and subsistence farmers (63). These codes
# usually represent the operators, managers, and owners of farms.We assume
# they live on-site in their own private housing, such as a single-family home
# on a farm. We do not exclude ISCO code 92 (Agricultural, forestry, and
# fishinq workers) as they may be living in poor quality high-density housing.
# For example, seasonal farm workers in the US had high rates of COVID-19
# infection while living in on-site dorms. They would need safer housing in
# a future pandemic.
ONSITE_HOUSING_EXCLUDED_ISCO_L2 = ["61", "63"]

SIMILAR_COUNTRIES = (
    pd.read_csv(ESSENTIAL_WORKERS_DATA / "similar_countries.csv")
    .groupby("country_code")["similar_country_code"]
    .apply(list)
    .to_dict()
)


# Per-group ISIC × ISCO overlap factors derived from ILO WESO 2023
# Figure A1 (Annex). For each occupational group g, the group_overlap_g row of
# parameters_essential_workers.csv is the *global aggregate* fraction of
# workers in that ISCO occupational group who are also employed in a key
# (essential) ISIC industry.
#
# These factors are the dominant simplification in this pipeline: the
# correct calculation would intersect each country's ISCO × ISIC
# cross-tabulation, but we lack worker-level microdata, so we assume the
# overlap structure is the same in every country. This is sometimes
# materially wrong (e.g. the "Manual" overlap is much higher in
# agrarian economies because more manual workers are in essential
# agriculture). See the README for the full rationale and
# ``essential_workers_validation.py`` for the per-country deviation it produces.
#
# Source: ILO 2023, "The value of essential work", Figure A1 (Annex):
#   https://www.ilo.org/sites/default/files/wcmsp5/groups/public/@dgreports/@dcomm/@publ/documents/publication/wcms_871016.pdf
#
# ArmedForces sits outside ILO Figure A1 (the report excludes uniformed
# services from its headline global figures). We retain it with the
# Blueprint Biosecurity placeholder of 0.40 so downstream
# indoor-essential counts include armed forces; treat that figure as
# a low-confidence assumption rather than an ILO-derived number.
def group_overlaps(parameters):
    """
    Global overlap of each occupational group with essential industries.

    Arguments:
        parameters (dict): Fixed essential-worker parameters.

    Returns:
        dict: Group to overlap, in GROUPS order.
    """
    return {group: parameters[f"group_overlap_{group.lower()}"] for group in GROUPS}


def apply_group_overlaps(weights_template, overlaps):
    """
    ISCO level-2 weights with one set of group overlaps applied.

    Arguments:
        weights_template (pandas.DataFrame): Output of
            preprocessing.build_isco_lvl2_template.
        overlaps (dict): Group to overlap.

    Returns:
        pandas.DataFrame: The template with Group Overlap and the four weight
            columns in WEIGHT_COLUMNS.
    """
    weights = weights_template.copy()
    weights["Group Overlap"] = weights["Group"].map(overlaps).fillna(0.0)
    indoor = weights["indoors_context"]
    weights["ISCO_08_PollWeights"] = (
        weights["Vital Weight POLL"] * indoor * weights["Group Overlap"]
    )
    weights["ISCO_08_ILOWeights"] = (
        weights["Essential Weight ILO"] * indoor * weights["Group Overlap"]
    )
    weights["ISCO_08_PollWeights_Total"] = (
        weights["Vital Weight POLL"] * weights["Group Overlap"]
    )
    weights["ISCO_08_ILOWeights_Total"] = (
        weights["Essential Weight ILO"] * weights["Group Overlap"]
    )
    return weights


def worker_shares(employment, weights_template, overlaps_by_country, default_overlaps):
    """
    Share of each country's employment in each worker category and group.

    Employment not elsewhere classified (NEC) is shared out in proportion to
    the coded employment, so it carries the country's average weights.

    Arguments:
        employment (dict): Country to {ISCO code, "Tot" or "Not": employment}.
        weights_template (pandas.DataFrame): ISCO level-2 weights.
        overlaps_by_country (dict): Country to group overlaps.
        default_overlaps (dict): Overlaps for countries not in
            overlaps_by_country.

    Returns:
        tuple: (DataFrame with Country Name, occupational_group and the share of
            total employment in each of WORKER_COLUMNS, and DataFrame with
            Country Name and the essential and vital shares in the on-site
            housing excluded codes, before NEC is shared out).
    """
    weight_columns = list(WEIGHT_COLUMNS.values())
    group_rows, onsite_rows = [], []
    for country, codes in employment.items():
        total = codes.get("Tot")
        if not (total and total > 0):
            continue
        weights = apply_group_overlaps(
            weights_template, overlaps_by_country.get(country, default_overlaps)
        )
        coded = pd.Series(
            {
                code: value
                for code, value in codes.items()
                if code not in ("Tot", "Not") and pd.notna(value)
            },
            dtype=float,
        )
        known = weights.index.intersection(coded.index)
        weighted = weights.loc[known, weight_columns].fillna(0.0).mul(coded[known], axis=0)

        nec = codes.get("Not")
        nec_scale = 1.0
        if nec is not None and pd.notna(nec) and nec > 0 and coded.sum() > 0:
            nec_scale += nec / coded.sum()
        by_group = (
            weighted.groupby(weights.loc[known, "Group"]).sum().reindex(GROUPS).fillna(0.0)
        )
        shares = by_group * nec_scale / total
        shares.columns = list(WEIGHT_COLUMNS)
        group_rows.append(
            shares.rename_axis("occupational_group").reset_index().assign(
                **{"Country Name": country}
            )
        )

        onsite = weighted.reindex(ONSITE_HOUSING_EXCLUDED_ISCO_L2).sum() / total
        onsite_rows.append(
            {
                "Country Name": country,
                "onsite_excluded_essential": onsite["ISCO_08_ILOWeights_Total"],
                "onsite_excluded_vital": onsite["ISCO_08_PollWeights_Total"],
            }
        )
    by_group = pd.concat(group_rows, ignore_index=True)
    return (
        by_group[["Country Name", "occupational_group", *WORKER_COLUMNS]],
        pd.DataFrame(onsite_rows),
    )


def country_shares(group_shares):
    """
    Add up group shares to each country's percentage columns.

    Arguments:
        group_shares (pandas.DataFrame): First output of worker_shares.

    Returns:
        pandas.DataFrame: Country Name and PCT_COLUMNS, as fractions.
    """
    totals = group_shares.groupby("Country Name", sort=False)[WORKER_COLUMNS].sum()
    totals.columns = PCT_COLUMNS
    return totals.reset_index()


def essential_mass_by_group(codes, weights_template):
    """
    Employment in ILO essential occupations, by occupational group.

    Arguments:
        codes (dict): {ISCO code: employment} for one country.
        weights_template (pandas.DataFrame): ISCO level-2 weights.

    Returns:
        dict: Group to employment, before any overlap is applied.
    """
    masses = {group: 0.0 for group in GROUPS}
    for code, value in codes.items():
        if code == "Tot" or not pd.notna(value) or code not in weights_template.index:
            continue
        if weights_template.at[code, "Essential Weight ILO"] == 1:
            group = weights_template.at[code, "Group"]
            if group in masses:
                masses[group] += float(value)
    return masses


def essential_mass_at_overlaps(masses, overlaps):
    """
    Essential employment once overlaps are applied: the sum of overlap x mass.

    Arguments:
        masses (dict): Output of essential_mass_by_group.
        overlaps (dict): Group to overlap.

    Returns:
        float: Essential employment.
    """
    return sum(
        float(overlaps.get(group, 0.0)) * float(masses.get(group, 0.0))
        for group in GROUPS
    )


def calibrate_group_overlaps(masses, target, baseline, tol=1e-6):
    """
    Scale one country's overlaps so its essential employment hits a target.

    Every group except the armed forces moves by the same fraction x of its
    headroom: towards 1 when raising, towards 0 when lowering.

    Arguments:
        masses (dict): Output of essential_mass_by_group.
        target (float): Essential employment to match.
        baseline (dict): Global overlaps to start from.
        tol (float): Relative tolerance on hitting the target.

    Returns:
        tuple: (overlaps, x, direction, status). direction is "raise",
            "lower" or "none". status is "ok", "exact_at_baseline", or
            "infeasible_clipped" when x had to be held to 0 to 1.
    """
    start = {group: float(baseline[group]) for group in GROUPS}
    start_mass = essential_mass_at_overlaps(masses, start)
    target = float(target)

    if target <= 0:
        overlaps = {group: 0.0 for group in CALIBRATABLE_GROUPS}
        return {**overlaps, "ArmedForces": start["ArmedForces"]}, 1.0, "lower", "ok"
    if abs(start_mass - target) <= tol * max(target, 1.0):
        return start, 0.0, "none", "exact_at_baseline"

    if start_mass < target:
        direction = "raise"
        headroom = {group: 1.0 - start[group] for group in CALIBRATABLE_GROUPS}
    else:
        direction = "lower"
        headroom = {group: start[group] for group in CALIBRATABLE_GROUPS}
    denominator = sum(headroom[g] * masses.get(g, 0.0) for g in CALIBRATABLE_GROUPS)
    x_raw = 1.0 if denominator <= 0 else abs(target - start_mass) / denominator
    x = float(np.clip(x_raw, 0.0, 1.0))
    sign = 1.0 if direction == "raise" else -1.0
    overlaps = {
        group: start[group] + sign * x * headroom[group] for group in CALIBRATABLE_GROUPS
    }
    overlaps["ArmedForces"] = start["ArmedForces"]
    overlaps = {group: overlaps[group] for group in GROUPS}

    reached = essential_mass_at_overlaps(masses, overlaps)
    if abs(x_raw - x) > 1e-5 or abs(reached - target) > tol * max(target, 1.0):
        return overlaps, x, direction, "infeasible_clipped"
    return overlaps, x, direction, "ok"


def calibrate_overlaps(labour_force, employment, ilo_published, weights_template, baseline):
    """
    Calibrated group overlaps for every country in the labour-force table.

    Countries with ILO occupation data and a published ILO share are
    calibrated directly. The rest take the mean of similar countries, and any
    left after that take the global overlaps.

    Arguments:
        labour_force (pandas.DataFrame): Country Name and Country Code.
        employment (dict): Country to {ISCO code: employment}.
        ilo_published (pandas.DataFrame): Output of load_ilo_published_pct.
        weights_template (pandas.DataFrame): ISCO level-2 weights.
        baseline (dict): Global overlaps.

    Returns:
        pandas.DataFrame: One row per country with OVERLAP_COLUMNS,
            calibration_x, calibration_direction, solver_status,
            overlap_source, model_essential_mass and ilo_target_mass.
    """
    ilo_lookup = ilo_published.set_index("Country Name")[
        "ILO %essential (published)"
    ].to_dict()
    rows = []
    for country, code in zip(labour_force["Country Name"], labour_force["Country Code"]):
        row = {"Country Name": country, "Country Code": code}
        row.update({column: np.nan for column in OVERLAP_COLUMNS})
        row.update(
            calibration_x=np.nan,
            calibration_direction="",
            solver_status="",
            overlap_source="",
        )
        codes = employment.get(country)
        ilo_pct = ilo_lookup.get(country)
        total = codes.get("Tot") if codes else None
        if ilo_pct is not None and pd.notna(ilo_pct) and total and total > 0:
            masses = essential_mass_by_group(codes, weights_template)
            target = ilo_pct / 100.0 * total
            overlaps, x, direction, status = calibrate_group_overlaps(
                masses, target, baseline
            )
            row.update({f"overlap_{g}": overlaps[g] for g in CALIBRATABLE_GROUPS})
            row.update(
                calibration_x=x,
                calibration_direction=direction,
                solver_status=status,
                overlap_source=OVERLAP_SOURCE_ILO,
                model_essential_mass=essential_mass_at_overlaps(masses, baseline),
                ilo_target_mass=target,
            )
        rows.append(row)
    table = pd.DataFrame(rows)

    # Fill from similar countries that were calibrated or filled themselves.
    # Only neighbours calibrated directly mark the country as neighbour-filled.
    position = {code: i for i, code in reversed(list(enumerate(table["Country Code"])))}
    for _ in range(50):
        still_missing = False
        for i in table.index[table[OVERLAP_COLUMNS[0]].isna()]:
            neighbours = [
                position[code]
                for code in SIMILAR_COUNTRIES.get(table.at[i, "Country Code"], [])
                if code in position
            ]
            donors = [
                n
                for n in neighbours
                if table.at[n, "overlap_source"]
                in (OVERLAP_SOURCE_ILO, OVERLAP_SOURCE_NEIGHBOUR)
            ]
            if not donors:
                still_missing = True
                continue
            table.loc[i, OVERLAP_COLUMNS] = table.loc[donors, OVERLAP_COLUMNS].mean()
            if any(table.at[n, "overlap_source"] == OVERLAP_SOURCE_ILO for n in neighbours):
                table.at[i, "overlap_source"] = OVERLAP_SOURCE_NEIGHBOUR
            x_values = table.loc[neighbours, "calibration_x"].dropna()
            if len(x_values):
                table.at[i, "calibration_x"] = x_values.mean()
        if not still_missing:
            break

    unfilled = table[OVERLAP_COLUMNS[0]].isna()
    for group in CALIBRATABLE_GROUPS:
        table.loc[unfilled, f"overlap_{group}"] = baseline[group]
    table.loc[unfilled, ["overlap_source", "solver_status"]] = OVERLAP_SOURCE_GLOBAL
    table.loc[unfilled, "calibration_x"] = 0.0
    return table


def overlaps_by_country(calibration, baseline):
    """
    Turn the calibration table into one overlap dict per country.

    Arguments:
        calibration (pandas.DataFrame): Output of calibrate_overlaps.
        baseline (dict): Global overlaps, which supply the armed forces value.

    Returns:
        dict: Country name to {group: overlap}.
    """
    return {
        row["Country Name"]: {
            **{group: row[f"overlap_{group}"] for group in CALIBRATABLE_GROUPS},
            "ArmedForces": baseline["ArmedForces"],
        }
        for _, row in calibration.iterrows()
    }


def backfill_neighbours(df, similar_countries=None, cols=PCT_COLUMNS, max_iterations=50):
    """
    Fill missing values with the mean of similar countries.

    Each column is swept repeatedly, because a country's neighbours may only
    be filled in a later sweep. Values filled earlier in a sweep are used by
    later countries in the same sweep.

    Arguments:
        df (pandas.DataFrame): Table with a Country Code column.
        similar_countries (dict or None): Country code to similar country
            codes. Defaults to data/essential_workers/similar_countries.csv.
        cols (list): Columns to fill.
        max_iterations (int): Most sweeps per column.

    Returns:
        pandas.DataFrame: A copy with the gaps filled where possible.
    """
    if similar_countries is None:
        similar_countries = SIMILAR_COUNTRIES
    df = df.copy()
    for col in cols:
        for _ in range(max_iterations):
            still_missing = False
            current = dict(zip(df["Country Code"][::-1], df[col][::-1]))
            for i in df.index[df[col].isna()]:
                code = df.at[i, "Country Code"]
                values = [
                    float(current[n])
                    for n in similar_countries.get(code, [])
                    if n in current and pd.notna(current[n])
                ]
                if values:
                    df.at[i, col] = current[code] = sum(values) / len(values)
                else:
                    still_missing = True
            if not still_missing:
                break
    return df


def fill_missing_labour_force(labour_force, employment):
    """
    Use ILO total employment where the World Bank has no labour force figure.

    Arguments:
        labour_force (pandas.DataFrame): Labour force by country.
        employment (dict): Country to {ISCO code, "Tot": employment}.

    Returns:
        pandas.DataFrame: A copy with the gaps filled, e.g. for Palestine.
    """
    df = labour_force.copy()
    for i in df.index[df[LABOUR_FORCE_COL].isna()]:
        total = employment.get(df.at[i, "Country Name"], {}).get("Tot")
        if total is not None and pd.notna(total) and total > 0:
            df.at[i, LABOUR_FORCE_COL] = float(total)
    return df


def onsite_housing_requirements(by_country):
    """
    Essential and vital workers who need housing near work.

    The essential and vital totals each lose only their own share of the
    excluded farm codes (ONSITE_HOUSING_EXCLUDED_ISCO_L2).

    Arguments:
        by_country (pandas.DataFrame): Country table with worker counts and
            the onsite_excluded_essential and onsite_excluded_vital shares.

    Returns:
        pandas.DataFrame: A Global row then one row per country.
    """
    out = by_country[["Country Name", "Country Code"]].copy()
    for workforce, share in [
        ("Essential Workers", "onsite_excluded_essential"),
        ("Vital Workers", "onsite_excluded_vital"),
    ]:
        out[f"{workforce} (Housing Requirement)"] = by_country[workforce] - (
            by_country[share].fillna(0.0) * by_country[LABOUR_FORCE_COL]
        )
    out[["Essential Workers", "Vital Workers"]] = by_country[
        ["Essential Workers", "Vital Workers"]
    ]
    totals = out.drop(columns=["Country Name", "Country Code"]).sum()
    global_row = {"Country Name": "Global", "Country Code": "GLOBAL", **totals}
    return pd.concat([pd.DataFrame([global_row]), out], ignore_index=True)


def add_scaled_eca(df):
    """
    Add the mean eCADR per indoor essential and vital worker.

    Arguments:
        df (pandas.DataFrame): Table with indoor worker counts and CADR totals.

    Returns:
        pandas.DataFrame: df with the two scaled ECA columns set.
    """
    for column, workers, cadr in [
        (SCALED_ECA_ESSENTIAL_COL, "Indoor Essential Workers", ecadr.INDOOR_ESSENTIAL_CADR_COL),
        (SCALED_ECA_VITAL_COL, "Indoor Vital Workers", ecadr.INDOOR_VITAL_CADR_COL),
    ]:
        df[column] = np.where(df[workers] > 0, df[cadr] / df[workers], np.nan)
    return df


def aggregate_by_region(by_country):
    """
    Add up country counts to UN regions and recompute the shares.

    Arguments:
        by_country (pandas.DataFrame): Country table.

    Returns:
        pandas.DataFrame: One row per region.
    """
    sums = [LABOUR_FORCE_COL, *WORKER_COLUMNS, *ecadr.CADR_COLUMNS]
    regional = by_country.groupby("Region")[sums].sum().reset_index()
    for workers, pct in zip(WORKER_COLUMNS, PCT_COLUMNS):
        regional[pct] = regional[workers] / regional[LABOUR_FORCE_COL]
    return add_scaled_eca(regional)


def group_composition(by_group, by=None):
    """
    Share of each workforce made up by each occupational group.

    Only countries with ILO occupation data are in by_group, so these are
    shares rather than counts.

    Arguments:
        by_group (pandas.DataFrame): Workers by country and group.
        by (str or None): None for the world, or "Region".

    Returns:
        pandas.DataFrame: One row per group (per region), with the four
            "% of ..." columns as fractions.
    """
    counts = ["Essential Workers", "Vital Workers", *WORKER_COLUMNS[:2]]
    scopes = [(None, by_group)] if by is None else by_group.groupby(by)
    parts = []
    for scope, rows in scopes:
        totals = rows.groupby("occupational_group")[counts].sum()
        shares = (totals / totals.sum()).reindex([g for g in GROUPS if g in totals.index])
        shares.columns = [f"% of {column}" for column in counts]
        shares = shares.reset_index()
        if by is not None:
            shares.insert(0, by, scope)
        parts.append(shares)
    return pd.concat(parts, ignore_index=True)


def estimate(
    parameters_file=ESSENTIAL_WORKERS_PARAMETERS,
    data_dir=ESSENTIAL_WORKERS_DATA,
    indoor_context_method=None,
):
    """
    Estimate essential and vital workers, and their filtration requirements.

    Arguments:
        parameters_file (Path): parameters_essential_workers.csv or a copy.
        data_dir (Path): Folder holding the essential-worker data files.
        indoor_context_method (str or None): Overrides the parameters file.

    Returns:
        dict: by_country, by_group, by_region, onsite_housing, ashrae_table
            and mask_table, plus the inputs and calibration used by
            essential_workers_validation.py (parameters, weights_template,
            employment, ilo_published, calibration and overlaps).
    """
    parameters, _ = read_parameters(parameters_file)
    method = indoor_context_method or parameters["indoor_context_method"]
    inputs = read_inputs(data_dir, method)
    employment = inputs["employment"]
    template = inputs["weights_template"]
    baseline = group_overlaps(parameters)

    labour_force = fill_missing_labour_force(inputs["labour_force"], employment)
    calibration = calibrate_overlaps(
        labour_force, employment, inputs["ilo_published"], template, baseline
    )
    overlaps = overlaps_by_country(calibration, baseline)
    group_shares, onsite = worker_shares(employment, template, overlaps, baseline)

    by_country = labour_force.merge(
        country_shares(group_shares), on="Country Name", how="left"
    ).merge(onsite, on="Country Name", how="left")
    by_country = backfill_neighbours(
        by_country,
        cols=[*PCT_COLUMNS, "onsite_excluded_essential", "onsite_excluded_vital"],
    )
    for workers, pct in zip(WORKER_COLUMNS, PCT_COLUMNS):
        by_country[workers] = by_country[pct] * by_country[LABOUR_FORCE_COL]
    onsite_housing = onsite_housing_requirements(by_country)

    rooms = ecadr.load_room_types(Path(data_dir))
    ids = ["Country Name", "Country Code", "Region", LABOUR_FORCE_COL]
    by_group = by_country.loc[by_country[LABOUR_FORCE_COL] > 0, ids].merge(
        group_shares, on="Country Name"
    )
    by_group[WORKER_COLUMNS] = by_group[WORKER_COLUMNS].mul(
        by_group.pop(LABOUR_FORCE_COL), axis=0
    )
    by_group = ecadr.add_requirements(by_group, rooms, parameters)[
        [
            *ids[:3],
            "occupational_group",
            "occupancy_group",
            "occupancy_category",
            "Essential Workers",
            "Vital Workers",
            *WORKER_COLUMNS[:2],
            ecadr.SCALED_ECA_COL,
            *ecadr.CADR_COLUMNS,
        ]
    ]

    cadr = by_group.groupby("Country Code", as_index=False)[ecadr.CADR_COLUMNS].sum()
    by_country = add_scaled_eca(by_country.merge(cadr, on="Country Code", how="left"))
    by_country = backfill_neighbours(
        by_country,
        cols=[SCALED_ECA_ESSENTIAL_COL, SCALED_ECA_VITAL_COL, *ecadr.CADR_COLUMNS],
    )
    by_country = by_country[
        [
            "Country Name",
            "Country Code",
            LABOUR_FORCE_COL,
            "Region",
            *PCT_COLUMNS,
            *WORKER_COLUMNS,
            *ecadr.CADR_COLUMNS,
            SCALED_ECA_ESSENTIAL_COL,
            SCALED_ECA_VITAL_COL,
        ]
    ]

    return {
        "by_country": by_country,
        "by_group": by_group,
        "by_region": aggregate_by_region(by_country),
        "onsite_housing": onsite_housing,
        "ashrae_table": ecadr.scaled_ashrae_table(rooms, parameters),
        "mask_table": ecadr.mask_efficiency_table(rooms, parameters),
        "parameters": parameters,
        "weights_template": template,
        "employment": employment,
        "ilo_published": inputs["ilo_published"],
        "calibration": calibration,
        "overlaps": overlaps,
    }


def write_results(results, results_dir=ESSENTIAL_WORKERS_RESULTS):
    """
    Write the core essential-worker results.

    Arguments:
        results (dict): Output of estimate.
        results_dir (Path): Folder to write to.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "essential_workers_by_country.csv": results["by_country"],
        "essential_workers_by_group.csv": results["by_group"],
        "essential_workers_by_region.csv": results["by_region"],
        "onsite_housing_worker_requirements.csv": results["onsite_housing"],
        "group_composition_global.csv": group_composition(results["by_group"]),
        "group_composition_by_region.csv": group_composition(
            results["by_group"], by="Region"
        ),
        "ashrae241_scaled_table1.csv": results["ashrae_table"],
        "ashrae241_scaled_by_mask_efficiency.csv": results["mask_table"],
    }
    for name, table in tables.items():
        table.to_csv(results_dir / name, index=False)
    print(f"Wrote {len(tables)} tables to {results_dir}")


if __name__ == "__main__":
    write_results(estimate())
