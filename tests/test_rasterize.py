"""Grid geometry, point binning and void filling."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import inclined_plane

from lidar_terrain_lab.rasterize import Grid, cell_statistic, fill_voids


def test_from_bounds_snaps_outward_to_whole_cells() -> None:
    grid = Grid.from_bounds(10.2, 20.7, 15.1, 23.0, res=1.0, crs="EPSG:32631")
    assert (grid.x0, grid.bounds[1]) == (10.0, 20.0)
    assert grid.shape == (3, 6)
    assert grid.bounds == (10.0, 20.0, 16.0, 23.0)


def test_from_bounds_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="positive"):
        Grid.from_bounds(0, 0, 1, 1, res=0)
    with pytest.raises(ValueError, match="empty"):
        Grid.from_bounds(0, 0, 0, 1, res=1)


def test_feet_grid_reports_meters() -> None:
    foot = 0.3048
    grid = Grid(0.0, 100.0, 10.0, 10, 10, meters_per_unit=foot)
    assert grid.res_m == pytest.approx(3.048)
    assert grid.cell_area_m2 == pytest.approx(3.048**2)
    assert grid.extent_m == pytest.approx((0.0, 30.48, 0.0, 30.48))


def test_flat_index_marks_points_off_the_grid(plane_grid: Grid) -> None:
    x = np.array([500_000.5, 500_019.9, 499_999.9, 500_020.0])
    y = np.array([1_029.5, 1_000.1, 1_010.0, 1_010.0])
    assert plane_grid.flat_index(x, y).tolist() == [0, 29 * 20 + 19, -1, -1]


def test_cell_statistics(plane_grid: Grid) -> None:
    x = np.array([500_000.2, 500_000.8, 500_001.5])
    y = np.array([1_029.5, 1_029.5, 1_029.5])
    z = np.array([3.0, 5.0, 7.0])
    index = plane_grid.flat_index(x, y)
    assert cell_statistic(plane_grid, index, z, "min")[0, 0] == 3.0
    assert cell_statistic(plane_grid, index, z, "max")[0, 0] == 5.0
    assert cell_statistic(plane_grid, index, z, "mean")[0, 0] == 4.0
    counts = cell_statistic(plane_grid, index, z, "count")
    assert counts[0, :2].tolist() == [2.0, 1.0]
    assert np.isnan(cell_statistic(plane_grid, index, z, "max")[5, 5])
    with pytest.raises(ValueError, match="unknown"):
        cell_statistic(plane_grid, index, z, "median")  # type: ignore[arg-type]


def test_fill_voids_recovers_a_plane_exactly() -> None:
    plane = inclined_plane(40, 40, 0.3, -0.2)
    holed = plane.copy()
    holed[10:22, 5:30] = np.nan
    # A plane is harmonic, so relaxation converges to it.
    assert np.abs(fill_voids(holed, max_iter=5000, tolerance=1e-7) - plane).max() < 1e-3


def test_fill_voids_edge_cases() -> None:
    full = np.ones((3, 3))
    assert np.array_equal(fill_voids(full), full)
    with pytest.raises(ValueError, match="no valid"):
        fill_voids(np.full((3, 3), np.nan))


def test_bilinear_sample_and_local_meters(plane_grid: Grid) -> None:
    plane = inclined_plane(30, 20, 0.5, 0.25)
    xs, ys = plane_grid.cell_centers()
    x = np.array([xs[3] + 0.5, xs[10]])
    y = np.array([ys[7], ys[2] - 0.5])
    expected = np.array([plane[7, 3] + 0.25, plane[2, 10] - 0.125])
    assert plane_grid.sample(plane, x, y) == pytest.approx(expected)
    u, v = plane_grid.to_local_meters(np.array([500_005.0]), np.array([1_001.0]))
    assert (u[0], v[0]) == (5.0, 1.0)
