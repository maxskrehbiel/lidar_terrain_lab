"""The raster grid every product shares, plus point-to-cell binning and void filling.

Row 0 is the north edge and column 0 the west edge, matching GeoTIFF convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from rasterio.transform import Affine
from scipy import ndimage

from ._types import BoolArray, FloatArray, IntArray

SQUARE_METERS_PER_ACRE = 4046.8564224
SQUARE_METERS_PER_HECTARE = 10_000.0

CellStat = Literal["min", "max", "mean", "count"]


@dataclass(frozen=True)
class Grid:
    """A north-up raster lattice in a projected coordinate reference system.

    Attributes:
        x0: West edge in CRS units.
        y1: North edge in CRS units.
        res: Cell size in CRS units.
        width: Number of columns.
        height: Number of rows.
        crs: CRS as WKT or an ``EPSG:`` string; ``None`` for an unreferenced grid.
        meters_per_unit: Length of one CRS unit in meters (1.0 for UTM, ~0.3048 for feet).
    """

    x0: float
    y1: float
    res: float
    width: int
    height: int
    crs: str | None = None
    meters_per_unit: float = 1.0

    @classmethod
    def from_bounds(
        cls,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        res: float,
        crs: str | None = None,
        meters_per_unit: float = 1.0,
    ) -> Grid:
        """Build a grid covering a bounding box, snapped outward to whole cells.

        Args:
            xmin: West bound in CRS units.
            ymin: South bound in CRS units.
            xmax: East bound in CRS units.
            ymax: North bound in CRS units.
            res: Cell size in CRS units.
            crs: CRS as WKT or ``EPSG:`` string.
            meters_per_unit: Length of one CRS unit in meters.

        Returns:
            A grid whose edges are multiples of ``res``.

        Raises:
            ValueError: If the box is empty or the resolution is not positive.
        """
        if res <= 0:
            raise ValueError(f"resolution must be positive, got {res}")
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"empty bounding box: {(xmin, ymin, xmax, ymax)}")
        # The tolerance keeps float noise in the division from adding a sliver column.
        x0 = math.floor(xmin / res + 1e-9) * res
        y0 = math.floor(ymin / res + 1e-9) * res
        width = max(1, math.ceil((xmax - x0) / res - 1e-9))
        height = max(1, math.ceil((ymax - y0) / res - 1e-9))
        return cls(x0, y0 + height * res, res, width, height, crs, meters_per_unit)

    @property
    def shape(self) -> tuple[int, int]:
        """Array shape as ``(rows, columns)``."""
        return (self.height, self.width)

    @property
    def transform(self) -> Affine:
        """Affine transform from (column, row) to (x, y) at cell corners."""
        return Affine(self.res, 0.0, self.x0, 0.0, -self.res, self.y1)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Bounds as ``(west, south, east, north)`` in CRS units."""
        return (
            self.x0,
            self.y1 - self.height * self.res,
            self.x0 + self.width * self.res,
            self.y1,
        )

    @property
    def res_m(self) -> float:
        """Cell size in meters."""
        return self.res * self.meters_per_unit

    @property
    def cell_area_m2(self) -> float:
        """Area of one cell in square meters."""
        return self.res_m**2

    @property
    def extent_m(self) -> tuple[float, float, float, float]:
        """Extent in meters relative to the south-west corner, as matplotlib expects."""
        return (0.0, self.width * self.res_m, 0.0, self.height * self.res_m)

    def flat_index(self, x: FloatArray, y: FloatArray) -> IntArray:
        """Return the flattened cell index of each point, or -1 when it falls outside.

        Args:
            x: Point x coordinates in CRS units.
            y: Point y coordinates in CRS units.

        Returns:
            ``row * width + column`` per point, -1 for points off the grid.
        """
        col = np.floor((x - self.x0) / self.res).astype(np.int64)
        row = np.floor((self.y1 - y) / self.res).astype(np.int64)
        inside = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
        return np.where(inside, row * self.width + col, -1).astype(np.int64)

    def cell_centers(self) -> tuple[FloatArray, FloatArray]:
        """Return 1-D arrays of column-center x and row-center y coordinates."""
        xs = self.x0 + (np.arange(self.width, dtype=np.float64) + 0.5) * self.res
        ys = self.y1 - (np.arange(self.height, dtype=np.float64) + 0.5) * self.res
        return xs, ys

    def sample(self, raster: FloatArray, x: FloatArray, y: FloatArray) -> FloatArray:
        """Bilinearly interpolate a raster at point locations.

        Args:
            raster: Array with this grid's shape and no NaNs.
            x: Point x coordinates in CRS units.
            y: Point y coordinates in CRS units.

        Returns:
            Interpolated values; points off the grid take the nearest edge value.
        """
        rows = (self.y1 - y) / self.res - 0.5
        cols = (x - self.x0) / self.res - 0.5
        values = ndimage.map_coordinates(raster, [rows, cols], order=1, mode="nearest")
        return np.asarray(values, dtype=np.float64)

    def to_local_meters(self, x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Convert CRS coordinates to meters east and north of the south-west corner."""
        west, south, _, _ = self.bounds
        return (x - west) * self.meters_per_unit, (y - south) * self.meters_per_unit


def cell_statistic(grid: Grid, index: IntArray, values: FloatArray, stat: CellStat) -> FloatArray:
    """Summarize point values per cell.

    Args:
        grid: Target grid.
        index: Flat cell index per point from :meth:`Grid.flat_index` (-1 is ignored).
        values: One value per point (ignored for ``count``).
        stat: ``min``, ``max``, ``mean`` or ``count``.

    Returns:
        Array with the grid's shape; cells without points are NaN (0 for ``count``).

    Raises:
        ValueError: For an unknown statistic.
    """
    size = grid.width * grid.height
    keep = index >= 0
    idx = index[keep]
    vals = values[keep]
    counts = np.bincount(idx, minlength=size).astype(np.float64)
    if stat == "count":
        return counts.reshape(grid.shape)
    if stat == "mean":
        sums = np.bincount(idx, weights=vals, minlength=size)
        out: FloatArray = np.divide(sums, counts, out=np.full(size, np.nan), where=counts > 0)
    elif stat == "min":
        out = np.full(size, np.inf)
        np.minimum.at(out, idx, vals)
    elif stat == "max":
        out = np.full(size, -np.inf)
        np.maximum.at(out, idx, vals)
    else:
        raise ValueError(f"unknown statistic {stat!r}")
    out[counts == 0] = np.nan
    return out.reshape(grid.shape)


def fill_voids(raster: FloatArray, max_iter: int = 400, tolerance: float = 1e-4) -> FloatArray:
    """Fill NaN cells with a smooth surface pinned to the surrounding valid cells.

    Voids start at their nearest valid value, then relax toward the average of their four
    neighbors (a discrete Laplace equation). The result is the smoothest surface that
    matches the known cells, which suits gaps under tree crowns and small water bodies.

    Args:
        raster: 2-D array with NaN marking voids.
        max_iter: Upper bound on relaxation sweeps.
        tolerance: Stop once no void cell moves more than this between sweeps.

    Returns:
        A float64 copy with every void filled.

    Raises:
        ValueError: If the raster has no valid cells at all.
    """
    void: BoolArray = ~np.isfinite(raster)
    if not void.any():
        return raster.astype(np.float64)
    if void.all():
        raise ValueError("cannot fill a raster that has no valid cells")
    nearest = ndimage.distance_transform_edt(void, return_distances=False, return_indices=True)
    filled: FloatArray = raster[tuple(nearest)].astype(np.float64)
    for _ in range(max_iter):
        padded = np.pad(filled, 1, mode="edge")
        neighbor_mean = 0.25 * (
            padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        )
        change = float(np.abs(neighbor_mean[void] - filled[void]).max())
        filled[void] = neighbor_mean[void]
        if change < tolerance:
            break
    return filled
