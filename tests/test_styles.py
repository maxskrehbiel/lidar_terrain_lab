"""Shared colors, ramps and PNG settings."""

from __future__ import annotations

import pytest
from matplotlib.colors import to_rgba

from lidar_terrain_lab.styles import CANOPY_RAMP, FLOW_RAMP, PNG_METADATA, ramp


@pytest.mark.parametrize("colors", [CANOPY_RAMP, FLOW_RAMP])
def test_ramp_runs_from_first_to_last_color(colors: tuple[str, ...]) -> None:
    cmap = ramp("test", colors)
    assert cmap(0.0) == pytest.approx(to_rgba(colors[0]))
    assert cmap(1.0) == pytest.approx(to_rgba(colors[-1]))


def test_png_metadata_drops_the_software_tag() -> None:
    assert PNG_METADATA == {"Software": None}
