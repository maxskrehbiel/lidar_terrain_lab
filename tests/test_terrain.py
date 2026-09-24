"""Slope, aspect and hillshade on surfaces with known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest
from helpers import inclined_plane

from lidar_terrain_lab.config import HillshadeParams
from lidar_terrain_lab.terrain import aspect_degrees, hillshade, horn_gradient, slope_degrees


def test_gradient_of_a_plane() -> None:
    dz_dx, dz_dy = horn_gradient(inclined_plane(9, 9, 0.2, -0.1), res_m=2.0)
    assert dz_dx[4, 4] == pytest.approx(0.1)
    assert dz_dy[4, 4] == pytest.approx(-0.05)


def test_slope_of_a_ten_percent_plane() -> None:
    slope = slope_degrees(inclined_plane(9, 9, 0.1, 0.0), res_m=1.0)
    assert slope[4, 4] == pytest.approx(math.degrees(math.atan(0.1)))


@pytest.mark.parametrize(
    ("dz_dx", "dz_dy", "facing"),
    [(1.0, 0.0, 270.0), (-1.0, 0.0, 90.0), (0.0, 1.0, 180.0), (0.0, -1.0, 0.0)],
)
def test_aspect_is_the_downhill_compass_direction(
    dz_dx: float, dz_dy: float, facing: float
) -> None:
    aspect = aspect_degrees(inclined_plane(9, 9, dz_dx, dz_dy), res_m=1.0)
    assert aspect[4, 4] % 360 == pytest.approx(facing)


def test_flat_ground_has_no_aspect() -> None:
    assert np.isnan(aspect_degrees(np.zeros((5, 5)), res_m=1.0)).all()


def test_hillshade_lights_slopes_facing_the_sun() -> None:
    sun = HillshadeParams(azimuth_deg=315.0, altitude_deg=45.0)
    # Elevation falling toward the west and north means the slope faces north-west.
    facing_nw = hillshade(inclined_plane(9, 9, 0.5, -0.5), 1.0, sun)[4, 4]
    facing_se = hillshade(inclined_plane(9, 9, -0.5, 0.5), 1.0, sun)[4, 4]
    flat = hillshade(np.zeros((9, 9)), 1.0, sun)[4, 4]
    assert facing_nw > flat > facing_se
    assert flat == pytest.approx(math.cos(math.radians(45.0)))
