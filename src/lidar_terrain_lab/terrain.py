"""Terrain derivatives from a DEM: slope, aspect and shaded relief (Horn's 3x3 method)."""

from __future__ import annotations

import numpy as np

from ._types import FloatArray
from .config import HillshadeParams

FLAT_GRADIENT = 1e-6  # rise over run below which aspect is undefined


def horn_gradient(dem: FloatArray, res_m: float) -> tuple[FloatArray, FloatArray]:
    """Rate of elevation change toward the east and toward the north.

    Horn (1981) weights the four direct neighbors twice as heavily as the diagonals,
    which damps noise better than a two-point difference. Edges repeat the border value.

    Args:
        dem: Elevations in meters, row 0 at the north.
        res_m: Cell size in meters.

    Returns:
        ``(dz_dx, dz_dy)``: rise per meter eastward and northward.
    """
    p = np.pad(dem, 1, mode="edge")
    nw, n, ne = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    w, e = p[1:-1, :-2], p[1:-1, 2:]
    sw, s, se = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
    dz_dx = ((ne + 2 * e + se) - (nw + 2 * w + sw)) / (8 * res_m)
    dz_dy = ((nw + 2 * n + ne) - (sw + 2 * s + se)) / (8 * res_m)
    return dz_dx, dz_dy


def slope_degrees(dem: FloatArray, res_m: float) -> FloatArray:
    """Steepness of the ground in degrees (0 = flat, 90 = vertical).

    ``slope = atan(sqrt(dz_dx**2 + dz_dy**2))``.

    Args:
        dem: Elevations in meters.
        res_m: Cell size in meters.

    Returns:
        Slope per cell in degrees.
    """
    dz_dx, dz_dy = horn_gradient(dem, res_m)
    return np.asarray(np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))), dtype=np.float64)


def aspect_degrees(dem: FloatArray, res_m: float) -> FloatArray:
    """Compass direction the slope faces (downhill), clockwise from north.

    A slope that drops toward the east has aspect 90. Flat cells are NaN.

    Args:
        dem: Elevations in meters.
        res_m: Cell size in meters.

    Returns:
        Aspect per cell in degrees on ``[0, 360)``.
    """
    dz_dx, dz_dy = horn_gradient(dem, res_m)
    aspect = np.degrees(np.arctan2(-dz_dx, -dz_dy)) % 360.0
    flat = np.hypot(dz_dx, dz_dy) < FLAT_GRADIENT
    return np.where(flat, np.nan, aspect)


def hillshade(dem: FloatArray, res_m: float, params: HillshadeParams) -> FloatArray:
    """Shaded relief: how brightly a low sun would light each cell.

    ``brightness = cos(zenith) cos(slope) + sin(zenith) sin(slope) cos(sun_azimuth - aspect)``,
    where zenith is 90 degrees minus the sun's altitude.

    Args:
        dem: Elevations in meters.
        res_m: Cell size in meters.
        params: Sun azimuth and altitude, and vertical exaggeration.

    Returns:
        Brightness on ``[0, 1]``.
    """
    dz_dx, dz_dy = horn_gradient(dem * params.z_factor, res_m)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dx, -dz_dy)
    zenith = np.radians(90.0 - params.altitude_deg)
    azimuth = np.radians(params.azimuth_deg)
    shade = np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(
        azimuth - aspect
    )
    return np.asarray(np.clip(shade, 0.0, 1.0), dtype=np.float64)
