"""Tests for the pathogen and mask scaling of ASHRAE-241 eCADR."""

import pandas as pd
import pytest

import ecadr_requirements as ecadr
from processing.paths import ESSENTIAL_WORKERS_PARAMETERS, read_parameters

PARAMETERS, _ = read_parameters(ESSENTIAL_WORKERS_PARAMETERS)
ROOMS = ecadr.load_room_types()


def _parameters(**overrides):
    """The real essential-worker parameters with some values replaced."""
    return {**PARAMETERS, **overrides}


def _food_room():
    """The Food row of the room table (ASHRAE-241 manufacturing)."""
    return ROOMS[ROOMS.occupational_group == "Food"]


@pytest.mark.parametrize("percentile", [5, 50, 96.3])
def test_sars_cov_2_against_itself_is_one(percentile):
    """The reference pathogen scales to itself at any percentile."""
    assert ecadr.scale_factor("SARS-CoV-2", percentile) == pytest.approx(1.0)


def test_measles_median_is_the_published_fold():
    """At the median, measles is the quanta-per-ml ratio of about 5.7."""
    assert ecadr.scale_factor("Measles", 50) == pytest.approx(5.68, abs=0.02)


def test_measles_at_ashrae_percentile():
    """At the 96.3th percentile, measles' wider spread raises the factor."""
    assert ecadr.scale_factor("Measles", 96.3) == pytest.approx(12.9, abs=0.05)


def test_relative_sd_is_wider_above_the_median():
    """Scaling ASHRAE's SD gives a wider measles spread than Mikszewski's own."""
    assert ecadr.scale_factor("Measles", 96.3, "relative") > ecadr.scale_factor(
        "Measles", 96.3, "absolute"
    )


def test_room_types_follow_the_group_order():
    """One room per occupational group, health care in a treatment area."""
    assert len(ROOMS) == 9
    health = ROOMS[ROOMS.occupational_group == "Health"].iloc[0]
    assert health.occupancy_category == "Group treatment area"


def test_manufacturing_room_matches_the_hand_calculation():
    """At QER 5.7 unmasked manufacturing needs 374 L/s/p; 30% masks give 158."""
    scaled = ecadr.scale_rooms(_food_room(), PARAMETERS, 5.7, 0.3, 0.0).iloc[0]
    assert scaled.each == pytest.approx(25 * 70 * 3.6 / 12000)
    assert scaled.k == pytest.approx(15.0, abs=0.05)
    assert round(scaled.scaled) == 374
    assert round(scaled.each * scaled.k, 1) == pytest.approx(7.9)

    masked = ecadr.scale_rooms(_food_room(), PARAMETERS, 5.7, 0.3, 0.3).iloc[0]
    assert round(masked.scaled) == 158


def test_outdoor_air_credit_is_taken_from_the_scaled_ecadr():
    """Net eCADR is the scaled value less the weighted outdoor airflow."""
    room = _food_room()
    scaled = ecadr.scale_rooms(room, PARAMETERS, 5.7, 0.3, 0.0).iloc[0]
    credit = (
        PARAMETERS["existing_airflow_weight"]
        * room.baseline_outdoor_airflow_ls_per_person.iloc[0]
    )
    assert scaled.net == pytest.approx(scaled.scaled - credit)


def test_health_care_masks_at_the_baseline_change_nothing():
    """ASHRAE-241 already assumes 30% masks in health care."""
    health = ROOMS[ROOMS.occupational_group == "Health"]
    parameters = _parameters(u_base_healthcare=0.3)
    masked = ecadr.scale_rooms(health, parameters, 5.7, 0.3, 0.0).iloc[0]
    plain = ecadr.scale_rooms(health, _parameters(u_base_healthcare=0.0), 5.7, 0.0, 0.0)
    assert masked.scaled == pytest.approx(plain.iloc[0].scaled)


def test_net_ecadr_is_never_negative():
    """Very good masks take the requirement to zero, not below it."""
    scaled = ecadr.scale_rooms(ROOMS, PARAMETERS, 5.7, 0.99, 0.99)
    assert (scaled.net >= 0).all()


def test_scaled_table_has_one_row_per_group():
    """The scaled ASHRAE table keeps Table 1's columns for every group."""
    table = ecadr.scaled_ashrae_table(ROOMS, PARAMETERS)
    assert list(table["Occupational group"]) == list(ROOMS.occupational_group)
    assert table["Scaled eCADR (L/s/p)"].dtype.kind == "i"


def test_better_masks_need_less_filtration():
    """Each step up in mask efficiency lowers every group's eCADR."""
    table = ecadr.mask_efficiency_table(ROOMS, PARAMETERS)
    masks = ecadr.mask_efficiencies(PARAMETERS)
    columns = [f"eCADR {mask:.0%} efficiency masks (L/s/p)" for mask in masks]
    assert (table[columns].diff(axis=1).iloc[:, 1:] <= 0).all().all()


def test_requirements_are_workers_times_net_ecadr():
    """Group requirements multiply indoor workers by the net eCADR per person."""
    workers = pd.DataFrame(
        {
            "occupational_group": ["Health", "Retail"],
            "Indoor Essential Workers": [600.0, 400.0],
            "Indoor Vital Workers": [600.0, 200.0],
        }
    )
    out = ecadr.add_requirements(workers, ROOMS, PARAMETERS)
    net = ecadr.scale_rooms(
        ROOMS,
        PARAMETERS,
        ecadr.qer_ratio(PARAMETERS),
        PARAMETERS["u_new_healthcare"],
        PARAMETERS["u_new_other"],
    ).net.set_axis(ROOMS.occupational_group)

    retail = out[out.occupational_group == "Retail"].iloc[0]
    assert retail[ecadr.SCALED_ECA_COL] == pytest.approx(net["Retail"])
    assert retail[ecadr.INDOOR_VITAL_CADR_COL] == pytest.approx(200 * net["Retail"])
    assert (
        out[ecadr.INDOOR_ESSENTIAL_CADR_COVID_COL] < out[ecadr.INDOOR_ESSENTIAL_CADR_COL]
    ).all()
