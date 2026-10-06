"""Pytest configuration for the InRoomAirFilterScaleUp test suite.

Essential-worker tests run on the real data in data/essential_workers/.
"""

import pytest

import essential_workers as ew
from processing.paths import ESSENTIAL_WORKERS_DATA


@pytest.fixture(scope="session")
def data_dir():
    """Essential-worker inputs in data/essential_workers/."""
    return ESSENTIAL_WORKERS_DATA


@pytest.fixture(scope="session")
def ew_outputs():
    """Run the essential-worker model once and share the results."""
    return ew.estimate()
