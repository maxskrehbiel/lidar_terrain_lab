"""Surface model (highest return per cell) and canopy height model (surface minus ground)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ._types import FloatArray, UInt8Array
from .config import CanopyParams
from .point_cloud import NOISE_CLASSES
from .rasterize import Grid, cell_statistic

# Buildings, water and bridge decks are real surfaces but not vegetation.
NON_VEGETATION_CLASSES = (6, 9, 17)


def surface_model(
    grid: Grid,
    x: FloatArray,
    y: FloatArray,
    z: FloatArray,
    classification: UInt8Array,
    dem: FloatArray,
) -> FloatArray:
    """Highest vegetation-eligible return per cell.

    Empty cells take the mean of their occupied neighbors, or the ground when none are
    occupied, so a missing return never creates a false canopy gap or a false crown.

    Args:
        grid: Output grid.
        x: Point x in CRS units.
        y: Point y in CRS units.
        z: Point heights in meters.
        classification: ASPRS class per point.
        dem: Bare-earth DEM on the same grid.

    Returns:
        Surface heights in meters, never below the DEM.
    """
    excluded = (*NOISE_CLASSES, *NON_VEGETATION_CLASSES)
    keep = ~np.isin(classification, excluded)
    highest = cell_statistic(grid, grid.flat_index(x[keep], y[keep]), z[keep], "max")
    void = ~np.isfinite(highest)
    if void.any():
        valid = (~void).astype(np.float64)
        kernel = np.ones((3, 3))
        total = ndimage.convolve(np.where(void, 0.0, highest), kernel, mode="nearest")
        count = ndimage.convolve(valid, kernel, mode="nearest")
        neighbor_mean = np.divide(total, count, out=np.full(grid.shape, np.nan), where=count > 0)
        highest = np.where(void, neighbor_mean, highest)
        highest = np.where(np.isfinite(highest), highest, dem)
    return np.maximum(highest, dem)


def canopy_height_model(dsm: FloatArray, dem: FloatArray, params: CanopyParams) -> FloatArray:
    """Height of whatever stands on the ground: surface minus bare earth.

    Two light clean-ups: implausibly tall spikes (unflagged birds or wires) and "pits"
    (a cell whose only returns slipped between branches) are replaced by the local median.

    Args:
        dsm: Surface model in meters.
        dem: Bare-earth DEM in meters.
        params: Canopy settings.

    Returns:
        Canopy height in meters, zero on open ground.
    """
    chm = np.clip(dsm - dem, 0.0, None)
    median = np.asarray(ndimage.median_filter(chm, size=3, mode="nearest"), dtype=np.float64)
    spikes = chm > params.max_height_m
    chm = np.where(spikes, np.minimum(median, params.max_height_m), chm)
    if params.pit_fill_m > 0:
        pits = (median - chm) > params.pit_fill_m
        chm = np.where(pits, median, chm)
    return chm
