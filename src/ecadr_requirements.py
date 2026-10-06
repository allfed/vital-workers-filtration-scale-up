"""
Filtration (eCADR) each indoor worker needs, scaled from ASHRAE Standard 241.

ASHRAE-241 sets an equivalent clean airflow per person (eCADR) for each room
type, sized for SARS-CoV-2. The steps below scale it to another pathogen:

  1. Viral load. log10 viral load is normal. SARS-CoV-2 uses ASHRAE-241's
     mean 7.0 and SD 1.4. Other pathogens shift that mean by log10 of their
     quanta per ml relative to SARS-CoV-2 (Mikszewski et al. 2022, Table 1,
     in data/essential_workers/pathogen_viral_loads.csv).
  2. Quanta emission ratio (QER). The ratio of the two viral loads at the same
     percentile. ASHRAE-241 is designed to the 96.3rd.
  3. Masks. The QER is multiplied by (1 - u_new)^2 / (1 - u_base)^2, where
     u_base is the mask efficiency ASHRAE-241 already assumes.
  4. Scaled eCADR. Holding risk at the ASHRAE-241 level in the Wells-Riley
     model multiplies eCADR by k = QER + (QER - 1)(deposition + decay) / eACH,
     where eACH = eCADR x occupants x 3.6 / room volume.
  5. Outdoor-air credit. A share of the room's baseline outdoor airflow is
     subtracted, and the result is never below zero.

The input is a table of workers by occupational group, so requirements can be
worked out for any set of worker counts.
"""

import numpy as np
import pandas as pd
from scipy import stats

from processing.paths import ESSENTIAL_WORKERS_DATA
from processing.preprocessing import GROUPS

GVL_MEAN, GVL_SD = 7.0, 1.4

# cv is measured in different units per pathogen, and ci converts it to quanta,
# so pathogens are compared on log10(cv * ci).
PATHOGENS = pd.read_csv(ESSENTIAL_WORKERS_DATA / "pathogen_viral_loads.csv").set_index(
    "pathogen"
)
PATHOGENS["gvl_shift"] = (PATHOGENS.log10_cv + np.log10(PATHOGENS.ci)) - (
    PATHOGENS.log10_cv["SARS-CoV-2"] + np.log10(PATHOGENS.ci["SARS-CoV-2"])
)
PATHOGENS["sd_ratio"] = PATHOGENS.cv_sd / PATHOGENS.cv_sd["SARS-CoV-2"]

SCALED_ECA_COL = "Scaled ECA (L/s/person)"
INDOOR_ESSENTIAL_CADR_COL = "Indoor Essential CADR Requirement (L/s)"
INDOOR_VITAL_CADR_COL = "Indoor Vital CADR Requirement (L/s)"
INDOOR_ESSENTIAL_CADR_COVID_COL = "Indoor Essential CADR Requirement COVID (L/s)"
INDOOR_VITAL_CADR_COVID_COL = "Indoor Vital CADR Requirement COVID (L/s)"
CADR_COLUMNS = [
    INDOOR_ESSENTIAL_CADR_COL,
    INDOOR_VITAL_CADR_COL,
    INDOOR_ESSENTIAL_CADR_COVID_COL,
    INDOOR_VITAL_CADR_COVID_COL,
]


def viral_load_distribution(pathogen, sd_mode="absolute"):
    """
    Viral load distribution of a pathogen, on Standard 241's scale.

    Arguments:
        pathogen (str): Name in PATHOGENS.
        sd_mode (str): "absolute" uses the pathogen's own log10 SD from
            Mikszewski et al. "relative" scales Standard 241's 1.4 by the
            pathogen's SD relative to SARS-CoV-2. SARS-CoV-2 uses 1.4 either way.

    Returns:
        scipy.stats frozen lognormal distribution of copies per ml.
    """
    p = PATHOGENS.loc[pathogen]
    if pathogen == "SARS-CoV-2":
        sd = GVL_SD
    elif sd_mode == "relative":
        sd = GVL_SD * p.sd_ratio
    elif sd_mode == "absolute":
        sd = p.cv_sd
    else:
        raise ValueError(f"sd_mode must be 'absolute' or 'relative', not {sd_mode}")
    return stats.lognorm(s=sd * np.log(10), scale=10 ** (GVL_MEAN + p.gvl_shift))


def scale_factor(pathogen, percentile, sd_mode="absolute"):
    """
    Ratio of a pathogen's viral load to SARS-CoV-2's at the same percentile.

    Arguments:
        pathogen (str): Name in PATHOGENS.
        percentile (float): Percentile to compare at, 0 to 100.
        sd_mode (str): "absolute" or "relative", see viral_load_distribution.

    Returns:
        float: The scale factor.
    """
    q = percentile / 100
    return float(
        viral_load_distribution(pathogen, sd_mode).ppf(q)
        / viral_load_distribution("SARS-CoV-2").ppf(q)
    )


def qer_ratio(parameters):
    """
    Quanta emission ratio of the chosen pathogen to SARS-CoV-2.

    Arguments:
        parameters (dict): Fixed essential-worker parameters.

    Returns:
        float: The ratio used to scale ASHRAE-241 eCADR.
    """
    return scale_factor(
        parameters["ashrae_pathogen"],
        parameters["viral_load_percentile"],
        parameters["viral_load_sd_mode"],
    )


def load_room_types(data_dir=ESSENTIAL_WORKERS_DATA):
    """
    The ASHRAE-241 room type assigned to each occupational group.

    Arguments:
        data_dir (Path): Folder with ashrae241_group_mapping.csv and
            ashrae241_eca_by_occupancy.csv.

    Returns:
        pandas.DataFrame: One row per group, in GROUPS order, with the room's
            eCADR per person, volume, occupants and baseline outdoor airflow.
    """
    rooms = pd.read_csv(data_dir / "ashrae241_group_mapping.csv").merge(
        pd.read_csv(data_dir / "ashrae241_eca_by_occupancy.csv"),
        on=["occupancy_group", "occupancy_category"],
        how="left",
    )
    needed = [
        "eca_ls_per_person",
        "space_vol",
        "max_occupants",
        "baseline_outdoor_airflow_ls_per_person",
    ]
    if set(GROUPS) - set(rooms.occupational_group) or rooms[needed].isna().any().any():
        raise ValueError("Every occupational group needs a complete ASHRAE room type")
    return rooms.set_index("occupational_group").loc[GROUPS].reset_index()


def scale_rooms(rooms, parameters, qer, mask_healthcare, mask_other):
    """
    Scale each room's ASHRAE-241 eCADR to the pathogen and masks given.

    Arguments:
        rooms (pandas.DataFrame): Output of load_room_types.
        parameters (dict): Fixed essential-worker parameters.
        qer (float): Quanta emission ratio before masks.
        mask_healthcare (float): Mask efficiency in health care.
        mask_other (float): Mask efficiency everywhere else.

    Returns:
        pandas.DataFrame: For each room, eACH (/h), k, scaled eCADR and net
            eCADR after the outdoor-air credit (L/s per person).
    """
    healthcare = rooms.occupancy_group == "Health care"
    mask_before = np.where(healthcare, parameters["u_base_healthcare"], 0.0)
    mask_after = np.where(healthcare, mask_healthcare, mask_other)
    qer = qer * (1 - mask_after) ** 2 / (1 - mask_before) ** 2
    each = rooms.eca_ls_per_person * rooms.max_occupants * 3.6 / rooms.space_vol
    removal = parameters["deposition_rate_per_hour"] + parameters["decay_rate_per_hour"]
    k = qer + (qer - 1) * removal / each
    scaled = rooms.eca_ls_per_person * k
    credit = (
        parameters["existing_airflow_weight"]
        * rooms.baseline_outdoor_airflow_ls_per_person
    )
    return pd.DataFrame(
        {"each": each, "k": k, "scaled": scaled, "net": (scaled - credit).clip(lower=0)}
    )


def scaled_ashrae_table(rooms, parameters, mask_healthcare=None, mask_other=None):
    """
    ASHRAE-241 eCADR by occupational group, scaled to the chosen pathogen.

    No outdoor-air credit is taken.

    Arguments:
        rooms (pandas.DataFrame): Output of load_room_types.
        parameters (dict): Fixed essential-worker parameters.
        mask_healthcare (float or None): Defaults to u_new_healthcare.
        mask_other (float or None): Defaults to u_new_other.

    Returns:
        pandas.DataFrame: One row per group.
    """
    if mask_healthcare is None:
        mask_healthcare = parameters["u_new_healthcare"]
    if mask_other is None:
        mask_other = parameters["u_new_other"]
    scaled = scale_rooms(
        rooms, parameters, qer_ratio(parameters), mask_healthcare, mask_other
    )
    return pd.DataFrame(
        {
            "Occupational group": rooms.occupational_group,
            "ASHRAE-241 room type": rooms.occupancy_category,
            "eCADR (L/s/p)": rooms.eca_ls_per_person.astype(float),
            "eACH (/h)": scaled.each.round(1),
            "Volume per occupant (m3)": (rooms.space_vol / rooms.max_occupants)
            .round()
            .astype(int),
            "k": scaled.k.round(1),
            "Scaled eCADR (L/s/p)": scaled.scaled.round().astype(int),
            "Scaled eACH (/h)": (scaled.each * scaled.k).round(1),
        }
    )


def mask_efficiency_table(rooms, parameters):
    """
    Scaled eCADR and eACH by occupational group at each mask efficiency.

    The same mask efficiency is used in health care and elsewhere. Where masks
    alone bring the risk below the ASHRAE-241 baseline the scaled value would
    be negative, so it is shown as zero.

    Arguments:
        rooms (pandas.DataFrame): Output of load_room_types.
        parameters (dict): Fixed essential-worker parameters.

    Returns:
        pandas.DataFrame: One row per group, with the ASHRAE-241 baseline and
            an eCADR and eACH column pair per mask efficiency.
    """
    table = scaled_ashrae_table(rooms, parameters)[
        ["Occupational group", "ASHRAE-241 room type", "eCADR (L/s/p)", "eACH (/h)"]
    ]
    for mask in mask_efficiencies(parameters):
        scaled = scaled_ashrae_table(rooms, parameters, mask, mask)
        label = f"{mask:.0%} efficiency masks"
        table[f"eCADR {label} (L/s/p)"] = scaled["Scaled eCADR (L/s/p)"].clip(lower=0)
        table[f"eACH {label} (/h)"] = scaled["Scaled eACH (/h)"].clip(lower=0)
    return table


def mask_efficiencies(parameters):
    """
    Mask efficiencies compared in the mask table and figure.

    Arguments:
        parameters (dict): Fixed essential-worker parameters.

    Returns:
        list: Mask efficiencies as floats.
    """
    return [float(mask) for mask in str(parameters["mask_efficiencies"]).split()]


def add_requirements(workers, rooms, parameters):
    """
    Add each group's room type, net eCADR per person and total requirements.

    The unlabelled requirement columns use the chosen pathogen and masks. The
    COVID columns leave ASHRAE-241 unscaled (covid_qer_ratio) with no masks
    outside health care (covid_u_new_other).

    Arguments:
        workers (pandas.DataFrame): Rows with occupational_group, Indoor
            Essential Workers and Indoor Vital Workers.
        rooms (pandas.DataFrame): Output of load_room_types.
        parameters (dict): Fixed essential-worker parameters.

    Returns:
        pandas.DataFrame: workers with occupancy_group, occupancy_category,
            SCALED_ECA_COL (L/s per person) and CADR_COLUMNS (L/s) added.
    """
    pathogen = scale_rooms(
        rooms,
        parameters,
        qer_ratio(parameters),
        parameters["u_new_healthcare"],
        parameters["u_new_other"],
    )
    covid = scale_rooms(
        rooms,
        parameters,
        parameters["covid_qer_ratio"],
        parameters["u_new_healthcare"],
        parameters["covid_u_new_other"],
    )
    per_group = rooms[["occupational_group", "occupancy_group", "occupancy_category"]]
    per_group = per_group.assign(**{SCALED_ECA_COL: pathogen.net, "covid": covid.net})
    out = workers.merge(per_group, on="occupational_group", how="left")
    for column, workforce, net in [
        (INDOOR_ESSENTIAL_CADR_COL, "Indoor Essential Workers", SCALED_ECA_COL),
        (INDOOR_VITAL_CADR_COL, "Indoor Vital Workers", SCALED_ECA_COL),
        (INDOOR_ESSENTIAL_CADR_COVID_COL, "Indoor Essential Workers", "covid"),
        (INDOOR_VITAL_CADR_COVID_COL, "Indoor Vital Workers", "covid"),
    ]:
        out[column] = out[workforce] * out[net]
    return out.drop(columns="covid")
