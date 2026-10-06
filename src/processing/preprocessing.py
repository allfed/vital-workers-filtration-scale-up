"""Load and transform the raw inputs of the essential-worker model."""

from __future__ import annotations

from pathlib import Path

import country_converter as coco
import numpy as np
import pandas as pd

_CC = coco.CountryConverter()

INDOOR_CONTEXT_METHODS = ("onet_max", "onet_banded", "jem_partial", "jem_binary")
INDOORS_CONTEXT_COLUMN = "indoors_context"

# ISCO-08 level-2 codes the ILO classes as essential (WESO 2023 Table A2), by
# the occupational group of ILO Figure A1. Armed forces are added on top.
GROUP_ISCO_CODES = {
    "Food": ["61", "62", "63", "92", "94"],
    "Health": ["22", "32", "53"],
    "Retail": ["52", "95"],
    "Security": ["54"],
    "Transport": ["83"],
    "Manual": ["71", "72", "73", "74", "75", "81", "82", "93"],
    "Cleaning": ["91", "96"],
    "Tech": ["31", "44", "51"],
    "ArmedForces": ["01", "02", "03"],
}
GROUPS = list(GROUP_ISCO_CODES)
ISCO_L2_TO_GROUP = {
    code: group for group, codes in GROUP_ISCO_CODES.items() for code in codes
}

# Codes in the vital-worker poll that the ILO does not class as essential.
NON_ILO_POLL_CODES = ["13", "21", "33", "35"]
# Armed-forces level-4 codes, which are always indoors.
OVERRIDES_INDOOR_L4 = ["0110", "0210", "0310"]

# Active-duty end strength from the DOD 2023 Demographics Profile (2024).
# Officers (Table 2.01) map to ISCO 01. Enlisted (1,038,909) is split by pay
# grade (Congressional Research Service IF10684, March 2024), scaled to the DOD
# enlisted total: E-5 through E-9 → ISCO 02, E-1 through E-4 → ISCO 03.
# Total active duty = 1,273,382.
US_ARMED_FORCES_EMPLOYMENT = {
    "01": 234_473,
    "02": 519_571,
    "03": 519_338,
}
ARMED_FORCES_ISCO_CODES = ("01", "02", "03")
SECURITY_ISCO_CODE = "54"
CLEANING_ISCO_CODES = ("91", "96")


def merge_onet_max_context(onet_controlled_df, onet_not_controlled_df):
    """
    Indoor percentage of each O*NET occupation.

    Arguments:
        onet_controlled_df (pandas.DataFrame): O*NET "indoors, environmentally
            controlled" context, with columns Code and Context.
        onet_not_controlled_df (pandas.DataFrame): O*NET "indoors, not
            environmentally controlled" context, same columns.

    Returns:
        pandas.DataFrame: Columns Code (SOC) and context_pct, the higher of the
            two percentages (0 to 100).
    """
    env = onet_controlled_df[["Code", "Context"]].copy()
    not_env = onet_not_controlled_df[["Code", "Context"]].copy()
    env["Code"] = env["Code"].map(lambda code: str(code).split(".")[0])
    not_env["Code"] = not_env["Code"].map(lambda code: str(code).split(".")[0])
    merged = env.merge(not_env, on="Code", how="outer", suffixes=("_env", "_not"))
    merged["context_pct"] = merged[["Context_env", "Context_not"]].max(axis=1)
    return merged[["Code", "context_pct"]]


def pct_to_indoor_fraction(context_pct, method):
    """
    Turn an O*NET indoor percentage into an indoor fraction.

    Arguments:
        context_pct (float): Indoor percentage, 0 to 100.
        method (str): "onet_max" divides by 100. "onet_banded" gives 1 from
            75%, 0.5 from 50% and 0 below.

    Returns:
        float: Indoor fraction, 0 to 1.
    """
    if method == "onet_max":
        return float(context_pct) / 100.0
    if context_pct >= 75:
        return 1.0
    if context_pct >= 50:
        return 0.5
    return 0.0


def location_to_indoor_fraction(location, partial=True):
    """
    Turn a job exposure matrix Location score into an indoor fraction.

    The score is rounded to the nearest of 0, 1, 2 and 3.

    Arguments:
        location (float): Location score, 0 (outdoors) to 3 (indoors).
        partial (bool): If True, 2 gives 0.5. If False, 2 and 3 both give 1.

    Returns:
        float: Indoor fraction, or NaN when the score is missing.
    """
    if partial:
        buckets = {0: 0.0, 1: 0.0, 2: 0.5, 3: 1.0}
    else:
        buckets = {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0}
    if pd.isna(location):
        return float("nan")
    nearest = min(buckets, key=lambda k: abs(float(location) - k))
    return buckets[nearest]


def load_jem_l4_indoor_fraction(jem_path, partial=True):
    """
    Indoor fraction of each ISCO level-4 code from the job exposure matrix.

    Arguments:
        jem_path (Path): Path to job_exposure_matrix.xls.
        partial (bool): Passed to location_to_indoor_fraction.

    Returns:
        dict: ISCO level-4 code to indoor fraction, from the mean Location
            across all country sheets.
    """
    sheets = pd.read_excel(jem_path, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    df["ISCO-08"] = df["ISCO-08"].astype(str).str.zfill(4)
    location = df.groupby("ISCO-08")["Location"].mean()
    return {
        str(code): location_to_indoor_fraction(value, partial=partial)
        for code, value in location.items()
    }


def onet_l4_indoor_fraction(
    onet_controlled_df, onet_not_controlled_df, crosswalk_df, indoor_context_method
):
    """
    Indoor fraction of each ISCO level-4 code from O*NET.

    Each SOC occupation is mapped through the crosswalk, and the ISCO code
    takes the mean of the SOC occupations mapped to it.

    Arguments:
        onet_controlled_df (pandas.DataFrame): See merge_onet_max_context.
        onet_not_controlled_df (pandas.DataFrame): See merge_onet_max_context.
        crosswalk_df (pandas.DataFrame): SOC 2010 to ISCO-08 crosswalk.
        indoor_context_method (str): "onet_max" or "onet_banded".

    Returns:
        dict: ISCO level-4 code to indoor fraction.
    """
    soc_df = merge_onet_max_context(onet_controlled_df, onet_not_controlled_df)
    crosswalk = crosswalk_df[["2010 SOC Code", "ISCO-08 Code"]].copy()
    crosswalk["ISCO-08 Code"] = crosswalk["ISCO-08 Code"].astype(str)
    soc_to_isco = crosswalk.set_index("2010 SOC Code")["ISCO-08 Code"].to_dict()

    fractions = {}
    for _, row in soc_df.iterrows():
        if row["Code"] in soc_to_isco:
            fractions.setdefault(soc_to_isco[row["Code"]], []).append(
                pct_to_indoor_fraction(row["context_pct"], indoor_context_method)
            )
    return {code: float(np.mean(values)) for code, values in fractions.items()}


def poll_l4_with_indoors_context(poll_df, l4_indoor):
    """
    Attach the indoor fraction to each ISCO level-4 row of the vital poll.

    Codes with no indoor fraction take the mean of their level-3, then level-2,
    then level-1 group. Armed forces are set fully indoors.

    Arguments:
        poll_df (pandas.DataFrame): Vital-worker poll, with columns ISCO-08
            and Census (the share of respondents rating the job vital).
        l4_indoor (dict): ISCO level-4 code to indoor fraction.

    Returns:
        pandas.DataFrame: Indexed by level-4 code, with Census and
            indoors_context.
    """
    df = poll_df[["ISCO-08", "Census"]].copy()
    df["ISCO-08"] = df["ISCO-08"].astype(str).str.zfill(4)
    df = df.set_index("ISCO-08")
    df["Census"] = pd.to_numeric(df["Census"], errors="coerce")
    df[INDOORS_CONTEXT_COLUMN] = df.index.map(l4_indoor)

    group_means = [
        df.groupby(df.index.str[:digits])[INDOORS_CONTEXT_COLUMN].transform("mean")
        for digits in (3, 2, 1)
    ]
    for group_mean in group_means:
        df[INDOORS_CONTEXT_COLUMN] = df[INDOORS_CONTEXT_COLUMN].fillna(group_mean)

    overrides = [code for code in OVERRIDES_INDOOR_L4 if code in df.index]
    df.loc[overrides, INDOORS_CONTEXT_COLUMN] = 1.0
    return df


def collapse_poll_l4_to_lvl2(l4_df):
    """
    Average the poll vital weight and indoor fraction to ISCO level 2.

    Market-oriented farmers (61) take the poll answers of 62, since the poll
    does not cover them. Subsistence farmers (63) are treated as fully
    outdoors. Poll codes the ILO does not class as essential are not vital.

    Arguments:
        l4_df (pandas.DataFrame): Output of poll_l4_with_indoors_context.

    Returns:
        pandas.DataFrame: Indexed by level-2 code, with Vital Weight POLL,
            indoors_context, Essential Weight ILO (1 or 0) and Group.
    """
    lvl2 = (
        l4_df.groupby(l4_df.index.str[:2])
        .agg({"Census": "mean", INDOORS_CONTEXT_COLUMN: "mean"})
        .rename_axis("ISCO-08")
        .rename(columns={"Census": "Vital Weight POLL"})
    )
    lvl2["Essential Weight ILO"] = lvl2.index.isin(list(ISCO_L2_TO_GROUP)).astype(int)
    if "62" in lvl2.index:
        lvl2.loc["61"] = lvl2.loc["62"]
    if "63" in lvl2.index:
        lvl2.at["63", INDOORS_CONTEXT_COLUMN] = 0.0
    for code in NON_ILO_POLL_CODES:
        if code in lvl2.index:
            lvl2.at[code, "Vital Weight POLL"] = 0
    lvl2["Group"] = lvl2.index.map(ISCO_L2_TO_GROUP)
    return lvl2


def build_isco_lvl2_template(
    poll_df,
    crosswalk_df,
    onet_controlled_df=None,
    onet_not_controlled_df=None,
    indoor_context_method="jem_binary",
    jem_path=None,
):
    """
    ISCO level-2 weights before the group overlaps are applied.

    Arguments:
        poll_df (pandas.DataFrame): Vital-worker poll.
        crosswalk_df (pandas.DataFrame): SOC 2010 to ISCO-08 crosswalk.
        onet_controlled_df (pandas.DataFrame or None): Needed for O*NET methods.
        onet_not_controlled_df (pandas.DataFrame or None): Needed for O*NET
            methods.
        indoor_context_method (str): One of INDOOR_CONTEXT_METHODS.
        jem_path (Path or None): Needed for the job exposure matrix methods.

    Returns:
        pandas.DataFrame: See collapse_poll_l4_to_lvl2.
    """
    if indoor_context_method in ("jem_partial", "jem_binary"):
        if jem_path is None:
            raise ValueError(f"jem_path is required for {indoor_context_method!r}")
        l4_indoor = load_jem_l4_indoor_fraction(
            jem_path, partial=(indoor_context_method == "jem_partial")
        )
    else:
        if onet_controlled_df is None or onet_not_controlled_df is None:
            raise ValueError(f"O*NET tables are required for {indoor_context_method!r}")
        l4_indoor = onet_l4_indoor_fraction(
            onet_controlled_df,
            onet_not_controlled_df,
            crosswalk_df,
            indoor_context_method,
        )
    return collapse_poll_l4_to_lvl2(poll_l4_with_indoors_context(poll_df, l4_indoor))


def _parse_isco_l2_code(classif_label):
    """ISCO-08 level-2 key from an ILO classif1.label: "22", "Tot" or "Not"."""
    part = str(classif_label).split(":", 1)[-1].strip()
    if part.lower().startswith("total"):
        return "Tot"
    if part.lower().startswith("not"):
        return "Not"
    return part[:2] if part[:2].isdigit() else part[:3].strip()


def _nec_share(year_codes):
    """Share of employment not elsewhere classified, or None without a total."""
    total = year_codes.get("Tot")
    nec = year_codes.get("Not")
    if total is None or not pd.notna(total) or total <= 0:
        return None
    if nec is None or not pd.notna(nec):
        return 0.0
    return float(nec) / float(total)


def _select_country_ilo_year(years):
    """
    Pick one ILO survey year for a country.

    Arguments:
        years (dict): Year to {ISCO code: employment}.

    Returns:
        int or None: The latest year with at most 10% of employment not
            elsewhere classified. If there is none, the year with the lowest
            such share, the later year winning ties.
    """
    if not years:
        return None
    sorted_years = sorted(years)
    if not any(_nec_share(years[y]) is not None for y in sorted_years):
        return sorted_years[-1]
    for year in reversed(sorted_years):
        share = _nec_share(years[year])
        if share is not None and share <= 0.10:
            return year
    best_year, best_share = sorted_years[0], float("inf")
    for year in sorted_years:
        share = _nec_share(years[year])
        if share is not None and (
            share < best_share or (share == best_share and year > best_year)
        ):
            best_year, best_share = year, share
    return best_year


def _coded_employment(employment):
    """Sum of positive ISCO level-2 employment, leaving out Tot and Not."""
    return sum(
        float(v)
        for k, v in employment.items()
        if k not in ("Tot", "Not") and pd.notna(v) and float(v) > 0
    )


def _isco_code_missing(employment, code):
    """True when an ISCO code is absent or has no usable employment."""
    value = employment.get(code)
    return value is None or not pd.notna(value) or float(value) <= 0


def _median_isco_shares(employment_by_country, codes):
    """
    Median share of coded employment for each code, across reporting countries.

    Arguments:
        employment_by_country (dict): Country to {ISCO code: employment}.
        codes (tuple): ISCO level-2 codes.

    Returns:
        dict: Code to median share, 0 where no country reports it.
    """
    shares = {code: [] for code in codes}
    for employment in employment_by_country.values():
        coded = _coded_employment(employment)
        if coded <= 0:
            continue
        for code in codes:
            if not _isco_code_missing(employment, code):
                shares[code].append(float(employment[code]) / coded)
    return {
        code: float(np.median(values)) if values else 0.0
        for code, values in shares.items()
    }


def impute_missing_ilo_employment(employment_by_country):
    """
    Fill missing ILO headcounts with global median shares of coded employment.

    Armed forces (01 to 03) and security (54) are filled for every country with
    a gap except the United States, which uses DOD active-duty end strength.
    Cleaning (91, 96) is filled for Laos only.

    Arguments:
        employment_by_country (dict): Country to {ISCO code: employment}.

    Returns:
        dict: The same dict, filled in place.
    """
    armed_shares = _median_isco_shares(employment_by_country, ARMED_FORCES_ISCO_CODES)
    security_share = _median_isco_shares(employment_by_country, (SECURITY_ISCO_CODE,))[
        SECURITY_ISCO_CODE
    ]
    cleaning_shares = _median_isco_shares(employment_by_country, CLEANING_ISCO_CODES)

    for country, employment in employment_by_country.items():
        coded = _coded_employment(employment)
        if coded <= 0:
            continue
        if country != "United States":
            for code in ARMED_FORCES_ISCO_CODES:
                if _isco_code_missing(employment, code) and armed_shares[code] > 0:
                    employment[code] = coded * armed_shares[code]
        if _isco_code_missing(employment, SECURITY_ISCO_CODE) and security_share > 0:
            employment[SECURITY_ISCO_CODE] = coded * security_share
        if country == "Laos":
            for code in CLEANING_ISCO_CODES:
                if _isco_code_missing(employment, code) and cleaning_shares[code] > 0:
                    employment[code] = coded * cleaning_shares[code]

    if "United States" in employment_by_country:
        for code in ARMED_FORCES_ISCO_CODES:
            employment_by_country["United States"][code] = float(
                US_ARMED_FORCES_EMPLOYMENT[code]
            )
    return employment_by_country


def build_employment_by_isco(ilo_df):
    """
    ILO employment by country and ISCO-08 level-2 code, in persons.

    Only the "Total" sex rows are kept. Each country uses one survey year (see
    _select_country_ilo_year), then gaps are filled by
    impute_missing_ilo_employment.

    Arguments:
        ilo_df (pandas.DataFrame): ILO ISCO-08 employment table, in thousands.

    Returns:
        dict: Country short name to {ISCO code, "Tot" or "Not": employment}.
    """
    df = ilo_df
    if "sex.label" in df.columns:
        df = df[df["sex.label"] == "Total"]
    df = pd.DataFrame(
        {
            "Country": df["ref_area.label"],
            "Code": df["classif1.label"].map(_parse_isco_l2_code),
            "Employment": pd.to_numeric(df["obs_value"], errors="coerce") * 1000,
            "time": df["time"],
        }
    ).dropna(subset=["Employment"])

    by_country_year = {}
    for row in df.itertuples():
        by_country_year.setdefault(row.Country, {}).setdefault(int(row.time), {})[
            row.Code
        ] = float(row.Employment)

    employment = {}
    for country, years in by_country_year.items():
        year = _select_country_ilo_year(years)
        if year is not None:
            name = _CC.convert(names=country, to="name_short", not_found="not found")
            employment[name] = dict(years[year])
    return impute_missing_ilo_employment(employment)


def load_ilo_published_pct(path):
    """
    ILO 2023 published share of essential workers in each country.

    Arguments:
        path (Path): ilo_country_essential_workers_pct.xlsx.

    Returns:
        pandas.DataFrame: Country Name, ILO %essential (published) and
            ILO %essential non-agri (published), in percent.
    """
    df = pd.read_excel(path, sheet_name="Sheet1", header=1, engine="openpyxl")
    df = df.rename(
        columns={
            "cname": "Country Name",
            "Share of key workers": "ILO %essential (published)",
            "Same share without agriculture": "ILO %essential non-agri (published)",
        }
    )
    columns = ["ILO %essential (published)", "ILO %essential non-agri (published)"]
    df = df[["Country Name", *columns]]
    df = df[df["Country Name"].notna() & (df["Country Name"] != "Average")].copy()
    df["Country Name"] = _CC.convert(
        df["Country Name"].tolist(), to="name_short", not_found="not found"
    )
    df[columns] = df[columns].apply(pd.to_numeric, errors="coerce")
    return df


def prepare_labour_force(lf_df):
    """
    Standardise country names in the labour-force table and add UN regions.

    Arguments:
        lf_df (pandas.DataFrame): World Bank labour force by country.

    Returns:
        pandas.DataFrame: The table with short country names and a Region.
    """
    df = lf_df.copy()
    df["Country Name"] = _CC.convert(
        df["Country Name"], to="short_name", not_found="not found"
    )
    df["Region"] = _CC.convert(df["Country Name"], to="UNregion", not_found="not found")
    return df


def read_inputs(data_dir, indoor_context_method):
    """
    Read every input of the essential-worker model.

    Arguments:
        data_dir (Path): Folder holding the essential-worker data files.
        indoor_context_method (str): One of INDOOR_CONTEXT_METHODS.

    Returns:
        dict: weights_template (ISCO level-2 weights before overlaps),
            employment (ILO employment by country and code), ilo_published
            (ILO published essential shares) and labour_force (labour force
            by country, with regions).
    """
    data_dir = Path(data_dir)
    onet = {}
    if indoor_context_method.startswith("onet"):
        onet = {
            "onet_controlled_df": pd.read_csv(
                data_dir / "onet_indoors_environmentally_controlled.csv"
            ),
            "onet_not_controlled_df": pd.read_csv(
                data_dir / "onet_indoors_not_environmentally_controlled.csv"
            ),
        }
    template = build_isco_lvl2_template(
        pd.read_excel(data_dir / "isco08_opinion_poll_census.xlsx", engine="openpyxl"),
        pd.read_csv(data_dir / "isco_soc_crosswalk.csv"),
        indoor_context_method=indoor_context_method,
        jem_path=data_dir / "job_exposure_matrix.xls",
        **onet,
    )
    return {
        "weights_template": template,
        "employment": build_employment_by_isco(
            pd.read_csv(data_dir / "ilo_isco08_employment.csv")
        ),
        "ilo_published": load_ilo_published_pct(
            data_dir / "ilo_country_essential_workers_pct.xlsx"
        ),
        "labour_force": prepare_labour_force(
            pd.read_excel(data_dir / "labour_force_wb_plus.xlsx", usecols=[0, 1, 3])
        ),
    }
