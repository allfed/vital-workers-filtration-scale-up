"""Tests for the viral-load scale factor."""

import pytest

import viral_load_scaler as vls


@pytest.mark.parametrize("percentile", [5, 50, 96.3])
def test_sars_cov_2_against_itself_is_one(percentile):
    """The reference pathogen scales to itself at any percentile."""
    assert vls.scale_factor("SARS-CoV-2", percentile) == pytest.approx(1.0)


def test_measles_median_is_the_published_fold():
    """At the median, measles is the quanta-per-ml ratio of about 5.7."""
    assert vls.scale_factor("Measles", 50) == pytest.approx(5.68, abs=0.02)


def test_measles_at_ashrae_percentile():
    """At the 96.3th percentile, measles' wider spread raises the factor."""
    assert vls.scale_factor("Measles", 96.3) == pytest.approx(12.9, abs=0.05)


def test_relative_sd_is_wider_above_the_median():
    """Scaling ASHRAE's SD gives a wider measles spread than Mikszewski's own."""
    assert vls.scale_factor("Measles", 96.3, "relative") > vls.scale_factor(
        "Measles", 96.3, "absolute"
    )
