"""Tests for the essential-worker model and its validation checks."""

import numpy as np
import pandas as pd
import pytest

import ecadr_requirements as ecadr
import essential_workers as ew
import essential_workers_validation as validation
import processing.preprocessing as pp
from processing.paths import ESSENTIAL_WORKERS_PARAMETERS, read_parameters

PARAMETERS, _ = read_parameters(ESSENTIAL_WORKERS_PARAMETERS)
BASELINE = ew.group_overlaps(PARAMETERS)
ILO_L2 = "Occupation (ISCO-08), 2 digit level:"


def _template(data_dir, method="onet_max"):
    """ISCO level-2 weights template from the O*NET tables in data_dir."""
    return pp.build_isco_lvl2_template(
        pd.read_excel(data_dir / "isco08_opinion_poll_census.xlsx"),
        pd.read_csv(data_dir / "isco_soc_crosswalk.csv"),
        pd.read_csv(data_dir / "onet_indoors_environmentally_controlled.csv"),
        pd.read_csv(data_dir / "onet_indoors_not_environmentally_controlled.csv"),
        indoor_context_method=method,
    )


def _ilo_row(country, classif, obs, year, sex="Total"):
    """One row of the ILO ISCO-08 employment table, in thousands."""
    return {
        "ref_area.label": country,
        "sex.label": sex,
        "classif1.label": classif,
        "time": year,
        "obs_value": obs,
    }


def _synthetic_template(codes):
    """Weights template for a few codes, fully indoors and all essential."""
    return pd.DataFrame(
        {
            "Group": {code: group for code, group, _ in codes},
            "Essential Weight ILO": {code: 1 for code, _, _ in codes},
            "Vital Weight POLL": {code: vital for code, _, vital in codes},
            pp.INDOORS_CONTEXT_COLUMN: {code: 1.0 for code, _, _ in codes},
        }
    )


# ---------------------------------------------------------------------------
# Indoor fraction and weights template
# ---------------------------------------------------------------------------


def test_onet_max_takes_the_higher_context():
    """Each O*NET occupation keeps the higher of its two indoor percentages."""
    controlled = pd.DataFrame({"Code": ["11-1111.00", "22-2222.00"], "Context": [50, 80]})
    not_controlled = pd.DataFrame(
        {"Code": ["11-1111.00", "22-2222.00"], "Context": [90, 40]}
    )
    merged = pp.merge_onet_max_context(controlled, not_controlled).set_index("Code")
    assert merged.at["11-1111", "context_pct"] == 90
    assert merged.at["22-2222", "context_pct"] == 80


def test_onet_banded_thresholds():
    """Banding gives 1 from 75%, 0.5 from 50% and 0 below."""
    assert pp.pct_to_indoor_fraction(80, "onet_banded") == 1.0
    assert pp.pct_to_indoor_fraction(60, "onet_banded") == 0.5
    assert pp.pct_to_indoor_fraction(40, "onet_banded") == 0.0
    assert pp.pct_to_indoor_fraction(80, "onet_max") == pytest.approx(0.8)


@pytest.mark.parametrize(
    "location, partial, expected",
    [(2.9, True, 1.0), (2.1, True, 0.5), (1.0, True, 0.0), (2.1, False, 1.0)],
)
def test_job_exposure_matrix_buckets(location, partial, expected):
    """Location scores round to the nearest bucket."""
    assert pp.location_to_indoor_fraction(location, partial) == expected


def test_indoor_method_does_not_change_total_weights(data_dir):
    """The indoor fraction only affects the indoor weights."""
    onet_max = ew.apply_group_overlaps(_template(data_dir, "onet_max"), BASELINE)
    banded = ew.apply_group_overlaps(_template(data_dir, "onet_banded"), BASELINE)
    for column in ["ISCO_08_PollWeights_Total", "ISCO_08_ILOWeights_Total"]:
        pd.testing.assert_series_equal(onet_max[column], banded[column])
    assert not onet_max["ISCO_08_ILOWeights"].equals(banded["ISCO_08_ILOWeights"])


def test_farm_codes_follow_the_special_cases(data_dir):
    """61 copies 62 from the poll, and subsistence farmers (63) are outdoors."""
    template = _template(data_dir)
    assert template.at["61", "Vital Weight POLL"] == template.at["62", "Vital Weight POLL"]
    assert template.at["63", pp.INDOORS_CONTEXT_COLUMN] == 0


def test_poll_codes_outside_ilo_essential_are_not_vital(data_dir):
    """Poll answers for codes the ILO does not class as essential are zeroed."""
    template = _template(data_dir)
    for code in pp.NON_ILO_POLL_CODES:
        assert template.at[code, "Vital Weight POLL"] == 0


def test_essential_weight_marks_the_ilo_groups(data_dir):
    """Every code mapped to a group is essential, and no other code is."""
    template = _template(data_dir)
    essential = set(template.index[template["Essential Weight ILO"] == 1])
    assert essential == set(pp.ISCO_L2_TO_GROUP) & set(template.index)


def test_group_overlaps_come_from_the_parameters(data_dir):
    """Each code takes its group's overlap from the parameters file."""
    weights = ew.apply_group_overlaps(_template(data_dir), BASELINE)
    for code, group in pp.ISCO_L2_TO_GROUP.items():
        if code in weights.index:
            assert weights.at[code, "Group Overlap"] == BASELINE[group]


# ---------------------------------------------------------------------------
# ILO employment
# ---------------------------------------------------------------------------


def test_employment_uses_the_latest_year_with_data():
    """NaN years are skipped."""
    health = f"{ILO_L2} 22 - Health"
    df = pd.DataFrame(
        [
            _ilo_row("Australia", health, 100.0, 2018),
            _ilo_row("Australia", health, 200.0, 2021),
            _ilo_row("Australia", health, np.nan, 2024),
        ]
    )
    assert pp.build_employment_by_isco(df)["Australia"]["22"] == 200_000.0


def test_employment_keeps_only_the_total_sex_rows():
    """Male and female rows are dropped."""
    total, health = f"{ILO_L2} Total", f"{ILO_L2} 22 - Health"
    df = pd.DataFrame(
        [
            _ilo_row("Barbados", total, 100.0, 2024, sex="Male"),
            _ilo_row("Barbados", total, 500.0, 2024),
            _ilo_row("Barbados", health, 999.0, 2024, sex="Female"),
            _ilo_row("Barbados", health, 50.0, 2024),
        ]
    )
    barbados = pp.build_employment_by_isco(df)["Barbados"]
    assert barbados["Tot"] == 500_000.0
    assert barbados["22"] == 50_000.0


def test_employment_skips_years_with_a_high_unclassified_share():
    """The latest year with at most 10% not elsewhere classified is used."""
    total, nec = f"{ILO_L2} Total", f"{ILO_L2} Not elsewhere classified"
    health = f"{ILO_L2} 22 - Health"
    df = pd.DataFrame(
        [
            _ilo_row("Belize", total, 1000.0, 2024),
            _ilo_row("Belize", nec, 800.0, 2024),
            _ilo_row("Belize", health, 50.0, 2024),
            _ilo_row("Belize", total, 1000.0, 2023),
            _ilo_row("Belize", nec, 50.0, 2023),
            _ilo_row("Belize", health, 100.0, 2023),
        ]
    )
    belize = pp.build_employment_by_isco(df)["Belize"]
    assert belize["Not"] == 50_000.0
    assert belize["22"] == 100_000.0


def test_employment_takes_the_lowest_unclassified_share_otherwise():
    """When every year is over 10% unclassified, the lowest share is used."""
    total, nec = f"{ILO_L2} Total", f"{ILO_L2} Not elsewhere classified"
    df = pd.DataFrame(
        [
            _ilo_row("Benin", total, 100.0, year)
            for year in (2022, 2023, 2024)
        ]
        + [
            _ilo_row("Benin", nec, value, year)
            for year, value in [(2022, 50.0), (2023, 40.0), (2024, 30.0)]
        ]
    )
    assert pp.build_employment_by_isco(df)["Benin"]["Not"] == 30_000.0


def test_missing_armed_forces_security_and_laos_cleaning_are_imputed():
    """Armed forces and security are filled everywhere, cleaning only in Laos."""
    rows = [
        ("01 - Commissioned armed forces officers", 10.0),
        ("02 - Non-commissioned armed forces officers", 20.0),
        ("03 - Armed forces occupations, other ranks", 30.0),
        ("54 - Protective services workers", 54.0),
        ("91 - Cleaners and helpers", 91.0),
        ("96 - Refuse workers and other elementary workers", 96.0),
        ("22 - Health professionals", 100.0),
    ]
    health = f"{ILO_L2} 22 - Health professionals"
    df = pd.DataFrame(
        [_ilo_row("Argentina", f"{ILO_L2} Total", 1000.0, 2024)]
        + [_ilo_row("Argentina", f"{ILO_L2} {label}", value, 2024) for label, value in rows]
        + [
            _ilo_row(country, f"{ILO_L2} Total", 500.0, 2024)
            for country in ["Belize", "Lao People's Democratic Republic", "Benin"]
        ]
        + [
            _ilo_row(country, health, value, 2024)
            for country, value in [
                ("Belize", 50.0),
                ("Lao People's Democratic Republic", 20.0),
                ("Benin", 30.0),
            ]
        ]
    )
    out = pp.build_employment_by_isco(df)
    argentina_coded = 401.0
    assert out["Belize"]["01"] == pytest.approx(50_000 * 10 / argentina_coded)
    assert out["Belize"]["54"] == pytest.approx(50_000 * 54 / argentina_coded)
    assert out["Laos"]["91"] == pytest.approx(20_000 * 91 / argentina_coded)
    assert "91" not in out["Benin"]


def test_us_armed_forces_use_the_defense_department_figures():
    """The US armed forces come from DOD end strength, not imputation."""
    df = pd.DataFrame(
        [
            _ilo_row("United States of America", f"{ILO_L2} Total", 150_000.0, 2024),
            _ilo_row("United States of America", f"{ILO_L2} 22 - Health", 50_000.0, 2024),
        ]
    )
    out = pp.build_employment_by_isco(df)
    for code, expected in pp.US_ARMED_FORCES_EMPLOYMENT.items():
        assert out["United States"][code] == expected


def test_labour_force_gaps_use_ilo_total_employment():
    """Palestine has no World Bank labour force, so ILO employment is used."""
    labour_force = pd.DataFrame(
        {
            "Country Name": ["Palestine", "Nigeria"],
            ew.LABOUR_FORCE_COL: [np.nan, 80_000_000.0],
        }
    )
    employment = {"Palestine": {"Tot": 719_891.0}, "Nigeria": {"Tot": 71_000_000.0}}
    out = ew.fill_missing_labour_force(labour_force, employment)
    assert list(out[ew.LABOUR_FORCE_COL]) == [719_891.0, 80_000_000.0]


# ---------------------------------------------------------------------------
# Worker shares
# ---------------------------------------------------------------------------


def test_unclassified_employment_carries_the_average_weights():
    """NEC employment scales each group by 1 + NEC / coded employment."""
    template = _synthetic_template([("52", "Retail", 0.5)])
    employment = {"Barbados": {"Tot": 1000.0, "Not": 200.0, "52": 800.0}}
    shares, _ = ew.worker_shares(employment, template, {}, BASELINE)
    retail = shares.set_index("occupational_group").loc["Retail"]
    overlap = BASELINE["Retail"]
    assert retail["Essential Workers"] == pytest.approx(800 * overlap * 1.25 / 1000)
    assert retail["Vital Workers"] == pytest.approx(800 * 0.5 * overlap * 1.25 / 1000)


def test_onsite_shares_count_codes_61_and_63_before_nec():
    """The on-site housing exclusion uses the coded farm employment only."""
    template = _synthetic_template([("61", "Food", 0.5), ("63", "Food", 1.0)])
    employment = {"Nigeria": {"Tot": 100.0, "Not": 10.0, "61": 10.0, "63": 30.0}}
    overlaps = {"Nigeria": {group: 1.0 for group in pp.GROUPS}}
    _, onsite = ew.worker_shares(employment, template, overlaps, BASELINE)
    assert onsite.at[0, "onsite_excluded_essential"] == pytest.approx(0.40)
    assert onsite.at[0, "onsite_excluded_vital"] == pytest.approx(0.35)


def test_neighbour_backfill_takes_the_mean_of_similar_countries():
    """A missing country takes the mean of its listed neighbours."""
    df = pd.DataFrame(
        {
            "Country Code": ["AAA", "BBB", "CCC", "ZZZ"],
            "%Essential Workers": [0.5, 0.6, 0.7, np.nan],
        }
    )
    out = ew.backfill_neighbours(
        df, {"ZZZ": ["AAA", "BBB", "CCC"]}, cols=["%Essential Workers"]
    )
    assert out.at[3, "%Essential Workers"] == pytest.approx(0.6)


def test_neighbour_backfill_reaches_countries_through_a_chain():
    """A country whose neighbour is also missing is filled once it is."""
    df = pd.DataFrame(
        {"Country Code": ["A", "B", "C"], "%Essential Workers": [0.5, np.nan, np.nan]}
    )
    out = ew.backfill_neighbours(
        df, {"C": ["B"], "B": ["A"]}, cols=["%Essential Workers"]
    )
    assert out["%Essential Workers"].tolist() == pytest.approx([0.5, 0.5, 0.5])


# ---------------------------------------------------------------------------
# Overlap calibration
# ---------------------------------------------------------------------------


def test_calibration_raises_overlaps_to_hit_the_target():
    """Raising moves every group but the armed forces towards 1."""
    masses = {group: 0.0 for group in pp.GROUPS} | {
        "Food": 100.0,
        "Manual": 50.0,
        "ArmedForces": 10.0,
    }
    target = ew.essential_mass_at_overlaps(masses, BASELINE) + 20.0
    overlaps, _, direction, _ = ew.calibrate_group_overlaps(masses, target, BASELINE)
    assert direction == "raise"
    assert ew.essential_mass_at_overlaps(masses, overlaps) == pytest.approx(target)
    assert overlaps["ArmedForces"] == BASELINE["ArmedForces"]


def test_calibration_lowers_overlaps_to_hit_the_target():
    """Lowering moves every group but the armed forces towards 0."""
    masses = {group: 100.0 for group in pp.GROUPS}
    target = ew.essential_mass_at_overlaps(masses, BASELINE) * 0.85
    overlaps, _, direction, _ = ew.calibrate_group_overlaps(masses, target, BASELINE)
    assert direction == "lower"
    assert ew.essential_mass_at_overlaps(masses, overlaps) == pytest.approx(target)


def test_unreachable_targets_are_clipped_and_flagged():
    """Overlaps stay between 0 and 1 when the target cannot be reached."""
    masses = {group: float(i + 1) for i, group in enumerate(pp.GROUPS)}
    overlaps, x, _, status = ew.calibrate_group_overlaps(masses, 500.0, BASELINE)
    assert status == "infeasible_clipped"
    assert x == 1.0
    assert all(0.0 <= overlaps[group] <= 1.0 for group in ew.CALIBRATABLE_GROUPS)


def test_every_country_gets_an_overlap(ew_outputs):
    """
    Calibrated, neighbour-filled or global, no overlap is left blank.

    A country filled only from neighbours that were themselves filled keeps a
    blank overlap_source.
    """
    calibration = ew_outputs["calibration"]
    assert calibration[ew.OVERLAP_COLUMNS].notna().all().all()
    assert set(calibration.overlap_source) <= {
        ew.OVERLAP_SOURCE_ILO,
        ew.OVERLAP_SOURCE_NEIGHBOUR,
        ew.OVERLAP_SOURCE_GLOBAL,
        "",
    }


# ---------------------------------------------------------------------------
# Results tables
# ---------------------------------------------------------------------------


def test_worker_counts_are_shares_times_labour_force(ew_outputs):
    """Counts are never negative and match share x labour force."""
    by_country = ew_outputs["by_country"]
    for workers, pct in zip(ew.WORKER_COLUMNS, ew.PCT_COLUMNS):
        assert (by_country[workers].dropna() >= 0).all()
        pd.testing.assert_series_equal(
            by_country[workers],
            by_country[pct] * by_country[ew.LABOUR_FORCE_COL],
            check_names=False,
        )


def test_indoor_workers_never_exceed_all_workers(ew_outputs):
    """Indoor essential and vital workers are subsets of the totals."""
    by_country = ew_outputs["by_country"].dropna(subset=["Essential Workers"])
    assert (by_country["Indoor Essential Workers"] <= by_country["Essential Workers"]).all()
    assert (by_country["Indoor Vital Workers"] <= by_country["Vital Workers"]).all()


def test_regions_add_up_their_countries(ew_outputs):
    """Regional totals are country sums, and shares are recomputed."""
    by_country, by_region = ew_outputs["by_country"], ew_outputs["by_region"]
    sums = by_country.groupby("Region")[["Essential Workers", *ecadr.CADR_COLUMNS]].sum()
    by_region = by_region.set_index("Region")
    pd.testing.assert_frame_equal(by_region[sums.columns], sums)
    pd.testing.assert_series_equal(
        by_region["%Essential Workers"],
        by_region["Essential Workers"] / by_region[ew.LABOUR_FORCE_COL],
        check_names=False,
    )


def test_onsite_housing_excludes_farm_owners(ew_outputs):
    """Housing needs are below the worker totals and the Global row adds up."""
    housing = ew_outputs["onsite_housing"]
    countries = housing[housing["Country Code"] != "GLOBAL"]
    global_row = housing[housing["Country Code"] == "GLOBAL"].iloc[0]
    for workforce in ["Essential Workers", "Vital Workers"]:
        column = f"{workforce} (Housing Requirement)"
        assert global_row[column] == pytest.approx(countries[column].sum())
        assert global_row[column] < global_row[workforce]
    nigeria = countries[countries["Country Name"] == "Nigeria"].iloc[0]
    assert 0 < nigeria["Essential Workers (Housing Requirement)"] < nigeria[
        "Essential Workers"
    ]
    assert len(housing) == len(ew_outputs["by_country"]) + 1


def test_country_requirements_add_up_their_groups(ew_outputs):
    """Country eCADR totals are the sum over groups, for ILO countries."""
    by_group, by_country = ew_outputs["by_group"], ew_outputs["by_country"]
    group_sums = by_group.groupby("Country Code")[ecadr.CADR_COLUMNS].sum()
    country = by_country.set_index("Country Code").loc[group_sums.index]
    pd.testing.assert_frame_equal(country[ecadr.CADR_COLUMNS], group_sums)
    pd.testing.assert_series_equal(
        country[ew.SCALED_ECA_ESSENTIAL_COL],
        country[ecadr.INDOOR_ESSENTIAL_CADR_COL] / country["Indoor Essential Workers"],
        check_names=False,
    )


def test_group_composition_shares_add_to_one(ew_outputs):
    """Each workforce is fully shared out between the groups."""
    composition = ew.group_composition(ew_outputs["by_group"])
    share_columns = [column for column in composition if column.startswith("% of")]
    assert composition[share_columns].sum().tolist() == pytest.approx([1.0] * 4)
    by_region = ew.group_composition(ew_outputs["by_group"], by="Region")
    region_sums = by_region.groupby("Region")[share_columns].sum()
    assert np.allclose(region_sums, 1.0)


def test_group_composition_is_worker_weighted():
    """Groups are weighted by workers, not averaged over countries."""
    by_group = pd.DataFrame(
        {
            "Country Code": ["AAA", "AAA", "BBB", "BBB"],
            "occupational_group": ["Food", "Health", "Food", "Health"],
            "Essential Workers": [80.0, 20.0, 20.0, 80.0],
            "Vital Workers": [40.0, 60.0, 10.0, 90.0],
            "Indoor Essential Workers": [10.0, 30.0, 5.0, 70.0],
            "Indoor Vital Workers": [5.0, 45.0, 2.0, 80.0],
        }
    )
    food = ew.group_composition(by_group).set_index("occupational_group").loc["Food"]
    assert food["% of Essential Workers"] == pytest.approx(0.5)
    assert food["% of Vital Workers"] == pytest.approx(0.25)


def test_write_results_writes_every_table(ew_outputs, tmp_path):
    """The eight core tables are written."""
    ew.write_results(ew_outputs, tmp_path)
    assert len(list(tmp_path.glob("*.csv"))) == 8


# ---------------------------------------------------------------------------
# Country-level checks
# ---------------------------------------------------------------------------


def test_every_country_is_filled(ew_outputs):
    """The neighbour backfill fills every country."""
    assert ew_outputs["by_country"][ew.PCT_COLUMNS].notna().all().all()


def test_global_essential_share_is_near_the_ilo(ew_outputs):
    """The global essential share is within 5pp of the ILO's 52%."""
    by_country = ew_outputs["by_country"]
    pct = 100 * by_country["Essential Workers"].sum() / by_country[ew.LABOUR_FORCE_COL].sum()
    assert abs(pct - 52.0) <= 5.0


def test_feasible_calibrations_hit_the_ilo_target(ew_outputs):
    """
    Where the target is reachable, calibrated overlaps give the ILO's share.

    The target is met on coded employment. Sharing out the not elsewhere
    classified employment afterwards moves the final share off it in
    countries with unclassified workers.
    """
    calibration = ew_outputs["calibration"]
    reached = calibration[calibration.solver_status == "ok"]
    for _, row in reached.iterrows():
        masses = ew.essential_mass_by_group(
            ew_outputs["employment"][row["Country Name"]],
            ew_outputs["weights_template"],
        )
        overlaps = ew_outputs["overlaps"][row["Country Name"]]
        assert ew.essential_mass_at_overlaps(masses, overlaps) == pytest.approx(
            row["ilo_target_mass"], rel=1e-6
        )


def test_liberia_is_flagged_as_unreachable(ew_outputs):
    """Liberia's ILO share needs overlaps above 1."""
    calibration = ew_outputs["calibration"].set_index("Country Name")
    assert calibration.at["Liberia", "solver_status"] == "infeasible_clipped"


def test_china_takes_its_neighbours_overlaps(ew_outputs):
    """China has no ILO occupation data, so it takes its neighbours' mean."""
    calibration = ew_outputs["calibration"].set_index("Country Code")
    assert calibration.at["CHN", "overlap_source"] == ew.OVERLAP_SOURCE_NEIGHBOUR
    donors = [
        code
        for code in ew.SIMILAR_COUNTRIES["CHN"]
        if calibration.at[code, "overlap_source"]
        in (ew.OVERLAP_SOURCE_ILO, ew.OVERLAP_SOURCE_NEIGHBOUR)
    ]
    assert calibration.at["CHN", "overlap_Food"] == pytest.approx(
        calibration.loc[donors, "overlap_Food"].mean()
    )


# ---------------------------------------------------------------------------
# Validation checks (src/essential_workers_validation.py)
# ---------------------------------------------------------------------------


def _by_group(rows):
    """Group table from (country code, group, essential, vital, indoor essential, indoor vital)."""
    return pd.DataFrame(
        rows,
        columns=[
            "Country Code",
            "occupational_group",
            "Essential Workers",
            "Vital Workers",
            "Indoor Essential Workers",
            "Indoor Vital Workers",
        ],
    )


def test_ilo_validation_table_has_both_shares(ew_outputs):
    """The validation table holds the model and calibrated shares and gaps."""
    result = validation.validate_against_ilo(ew_outputs)
    assert {
        "Our %Essential (model, global overlap)",
        "Our %Essential (calibrated)",
        "ILO %essential (published)",
        "Delta model (pp)",
        "Delta calibrated (pp)",
    } <= set(result["table"].columns)
    assert result["mean_abs_delta_pp"] >= 0


def test_calibration_detail_has_a_row_per_country_and_group(ew_outputs):
    """Every country gets one row for each occupational group."""
    detail = validation.calibration_detail(ew_outputs)
    assert len(detail) == 9 * len(ew_outputs["calibration"])


def test_outdoor_workers_are_total_minus_indoor(ew_outputs):
    """The global summary's outdoor rows are the residual of total and indoor."""
    by_country = ew_outputs["by_country"]
    summary = validation.global_worker_summary(by_country)
    assert summary.at["Outdoor essential workers", "Workers"] == pytest.approx(
        by_country["Essential Workers"].sum() - by_country["Indoor Essential Workers"].sum()
    )
    assert summary.at["Vital workers", "Country max %"] == pytest.approx(
        100 * by_country["%Vital Workers"].max()
    )


def test_food_share_of_each_workforce():
    """Food workers divided by each workforce's total."""
    by_group = _by_group(
        [("AAA", "Food", 80, 70, 20, 10), ("AAA", "Health", 20, 30, 20, 30)]
    )
    food = validation.food_share_of_workforce(by_group).iloc[0]
    assert food["Food % of Essential Workers"] == pytest.approx(0.8)
    assert food["Food % of Indoor Essential Workers"] == pytest.approx(0.5)
    assert food["Food % of Vital Workers"] == pytest.approx(0.7)
    assert food["Food % of Indoor Vital Workers"] == pytest.approx(0.25)


def test_richer_countries_have_smaller_essential_shares(tmp_path):
    """A synthetic falling share gives a negative rank correlation with GDP."""
    by_country = pd.DataFrame(
        {
            "Country Name": ["Poor", "Mid", "Rich"],
            "Country Code": ["POO", "MID", "RIC"],
            "Region": "R",
            "%Essential Workers": [0.7, 0.5, 0.3],
            "%Indoor Essential Workers": [0.25, 0.22, 0.20],
            "%Vital Workers": [0.6, 0.4, 0.2],
            "%Indoor Vital Workers": [0.18, 0.16, 0.14],
        }
    )
    by_group = _by_group(
        [
            ("POO", "Food", 90, 90, 10, 10),
            ("POO", "Health", 10, 10, 10, 10),
            ("MID", "Food", 50, 50, 20, 20),
            ("MID", "Health", 50, 50, 50, 50),
            ("RIC", "Food", 10, 10, 5, 5),
            ("RIC", "Health", 90, 90, 90, 90),
        ]
    )
    gdp_file = tmp_path / "gdp.csv"
    pd.DataFrame(
        {
            "Country Code": ["POO", "MID", "RIC"],
            validation.GDP_PPP_COL: [2000.0, 10000.0, 50000.0],
            "GDP Year (PPP)": 2024,
            "GDP Year (USD)": 2024,
            validation.GDP_USD_COL: [1000.0, 5000.0, 40000.0],
        }
    ).to_csv(gdp_file, index=False)

    _, summary = validation.worker_shares_vs_gdp(by_country, by_group, gdp_file)
    summary = summary.set_index("column")
    assert summary.at["%Essential Workers", "Spearman ρ"] < 0
    assert summary.at["Food % of Essential Workers", "Spearman ρ"] < 0


def test_pitman_morgan_detects_a_narrower_paired_series():
    """Halving one series' spread is detected, an equal spread is not."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    narrower = validation.pitman_morgan_variance_test(x, 0.5 * x + rng.normal(scale=0.1, size=500))
    assert narrower["variance_ratio_y_over_x"] < 0.5
    assert narrower["p_y_smaller"] < 0.001
    same = validation.pitman_morgan_variance_test(x, rng.permutation(x))
    assert same["variance_ratio_y_over_x"] == pytest.approx(1.0)
    assert same["p_two_sided"] > 0.05
