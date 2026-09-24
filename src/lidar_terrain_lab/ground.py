"""Ground classification (progressive morphological filter) and bare-earth DEM construction.

The filter follows Zhang et al. (2003), "A progressive morphological filter for removing
nonground measurements from airborne LIDAR data", IEEE TGRS 41(4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ._types import BoolArray, FloatArray
from .config import GroundParams
from .errors import InputDataError
from .point_cloud import GROUND_CLASS, PointCloud
from .rasterize import Grid, cell_statistic, fill_voids

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroundResult:
    """Outcome of ground classification.

    Attributes:
        is_ground: One flag per input point.
        method: ``existing`` or ``pmf``, whichever actually ran.
    """

    is_ground: BoolArray
    method: str


def window_sizes(cell_m: float, max_window_m: float) -> list[int]:
    """Odd window widths in cells, doubling from 3 until they exceed ``max_window_m``.

    Args:
        cell_m: Cell size in meters.
        max_window_m: Largest window in meters.

    Returns:
        Window widths ``3, 5, 9, 17, ...`` (always at least one).
    """
    sizes = [3]
    while (2 * sizes[-1] - 1) * cell_m <= max_window_m:
        sizes.append(2 * sizes[-1] - 1)
    return sizes


def pmf_ground_mask(
    x: FloatArray,
    y: FloatArray,
    z: FloatArray,
    params: GroundParams,
    cell_m: float = 1.0,
    meters_per_unit: float = 1.0,
) -> BoolArray:
    """Label ground points with a progressive morphological filter.

    The lowest point in each cell forms a rough surface. A morphological opening
    (erode, then dilate) with a small window shaves off anything narrower than the
    window that sticks up, such as shrubs; larger windows then remove tree crowns.
    A cell is non-ground when an opening lowers it by more than a height threshold
    that grows with the window, so broad hills survive while narrow objects do not.
    Points within ``tolerance_m`` of the surviving ground surface are ground.

    Args:
        x: Point x in CRS units.
        y: Point y in CRS units.
        z: Point heights in meters (noise already removed).
        params: Filter settings.
        cell_m: Working cell size in meters.
        meters_per_unit: Length of one CRS unit in meters.

    Returns:
        True for each ground point.
    """
    res = cell_m / meters_per_unit
    grid = Grid.from_bounds(
        float(x.min()), float(y.min()), float(x.max()) + 0.5 * res, float(y.max()) + 0.5 * res, res
    )
    index = grid.flat_index(x, y)
    lowest = cell_statistic(grid, index, z, "min")
    surface = fill_voids(lowest, max_iter=50)

    non_ground = np.zeros(grid.shape, dtype=bool)
    previous_window: int | None = None
    for window in window_sizes(cell_m, params.max_window_m):
        opened = np.asarray(ndimage.grey_opening(surface, size=(window, window)), dtype=np.float64)
        if previous_window is None:
            threshold = params.initial_threshold_m
        else:
            growth = params.terrain_slope * (window - previous_window) * cell_m
            threshold = min(params.initial_threshold_m + growth, params.max_threshold_m)
        non_ground |= (surface - opened) > threshold
        surface = opened
        previous_window = window

    ground_surface = fill_voids(np.where(non_ground, np.nan, lowest))
    height_above = z - grid.sample(ground_surface, x, y)
    return np.abs(height_above) <= params.tolerance_m


def classify_ground(cloud: PointCloud, params: GroundParams, cell_m: float) -> GroundResult:
    """Choose between the provider's ground class and the morphological filter.

    Args:
        cloud: Points with noise already removed.
        params: Ground settings, including the method.
        cell_m: Working cell size in meters.

    Returns:
        Per-point ground flags and the method used.

    Raises:
        InputDataError: If ``existing`` is requested but no point carries class 2.
    """
    provider_ground = cloud.classification == GROUND_CLASS
    share = float(provider_ground.mean()) if len(cloud) else 0.0
    method = params.method
    if method == "auto":
        method = "existing" if share >= params.min_existing_fraction else "pmf"
    if method == "existing":
        if not provider_ground.any():
            raise InputDataError("ground method 'existing' requested but no points have class 2")
        logger.info("using provider ground class (%.1f%% of points)", 100 * share)
        return GroundResult(provider_ground, "existing")
    logger.info("running progressive morphological filter")
    mask = pmf_ground_mask(cloud.x, cloud.y, cloud.z, params, cell_m, cloud.meters_per_unit)
    return GroundResult(mask, "pmf")


def bare_earth_dem(
    grid: Grid, x: FloatArray, y: FloatArray, z: FloatArray
) -> tuple[FloatArray, float]:
    """Average ground returns per cell and fill cells that received none.

    Args:
        grid: Output grid.
        x: Ground point x in CRS units.
        y: Ground point y in CRS units.
        z: Ground point heights in meters.

    Returns:
        The DEM and the fraction of cells that had to be interpolated.

    Raises:
        InputDataError: If no ground points fall on the grid.
    """
    mean = cell_statistic(grid, grid.flat_index(x, y), z, "mean")
    void_fraction = float((~np.isfinite(mean)).mean())
    if void_fraction == 1.0:
        raise InputDataError("no ground points fall inside the grid")
    return fill_voids(mean), void_fraction
