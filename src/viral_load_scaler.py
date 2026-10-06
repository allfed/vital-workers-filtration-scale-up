"""
Scale factor from SARS-CoV-2 to another pathogen, based on viral load.

Viral load is lognormal: log10 copies per ml is normal. SARS-CoV-2 uses the
distribution in ASHRAE Standard 241 (mean 7.0, SD 1.4 log10). Other pathogens
shift that mean by log10 of their quanta per ml relative to SARS-CoV-2, from
Mikszewski, Stabile, Buonanno & Morawska (2022), Geoscience Frontiers 13,
101285, Table 1. The scale factor is the ratio of the two distributions at the
same percentile, and is used as the quanta emission ratio in the ASHRAE-241
scale-up.
"""

import numpy as np
import pandas as pd
from scipy import stats

GVL_MEAN, GVL_SD = 7.0, 1.4

# cv is measured in different units per pathogen, and ci converts it to quanta,
# so pathogens are compared on log10(cv * ci).
PATHOGENS = pd.DataFrame(
    [
        ("SARS-CoV-2", 5.6, 1.2, 1.4e-3, "RNA copies/ml"),
        ("Measles", 3.5, 1.6, 1.0, "TCID50/ml"),
        ("Adenovirus", 3.2, 0.95, 0.50, "TCID50/ml"),
        ("TB (untreated)", 5.5, 1.3, 2.0e-3, "CFU/ml"),
        ("Rhinovirus", 3.6, 0.83, 0.053, "TCID50/ml"),
        ("Coxsackievirus", 3.4, 1.1, 0.025, "TCID50/ml"),
        ("Influenza", 6.7, 0.84, 7.1e-6, "RNA copies/ml"),
        ("MERS", 6.7, 1.6, 2.3e-6, "RNA copies/ml"),
        ("SARS-CoV-1", 6.1, 1.3, 6.8e-6, "RNA copies/ml"),
    ],
    columns=["pathogen", "log10_cv", "cv_sd", "ci", "cv_units"],
).set_index("pathogen")
PATHOGENS["gvl_shift"] = (PATHOGENS.log10_cv + np.log10(PATHOGENS.ci)) - (
    PATHOGENS.log10_cv["SARS-CoV-2"] + np.log10(PATHOGENS.ci["SARS-CoV-2"])
)
PATHOGENS["sd_ratio"] = PATHOGENS.cv_sd / PATHOGENS.cv_sd["SARS-CoV-2"]


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


if __name__ == "__main__":
    for mode in ["absolute", "relative"]:
        for percentile in [50, 96.3]:
            print(
                f"Measles, {mode} SD, percentile {percentile}: "
                f"{scale_factor('Measles', percentile, mode):.2f}"
            )
