"""Surface model and canopy height model."""

from __future__ import annotations

import numpy as np
import pytest

from lidar_terrain_lab.canopy import canopy_height_model, surface_model
from lidar_terrain_lab.config import CanopyParams
from lidar_terrain_lab.pipeline import Products
from lidar_terrain_lab.rasterize import Grid
from lidar_terrain_lab.synthetic import SMALL_TREE, TREE, SyntheticScene


def _points_at(grid: Grid, cells: list[tuple[int, int]], z: list[float], cls: list[int]):
    xs, ys = grid.cell_centers()
    x = np.array([xs[c] for _, c in cells])
    y = np.array([ys[r] for r, _ in cells])
    return x, y, np.array(z), np.array(cls, dtype=np.uint8)


def test_surface_ignores_noise_and_buildings(plane_grid: Grid) -> None:
    dem = np.zeros(plane_grid.shape)
    x, y, z, cls = _points_at(
        plane_grid,
        [(1, 1), (1, 1), (2, 2), (3, 3)],
        [4.0, 55.0, 9.0, 1.0],
        [1, 18, 6, 1],
    )
    dsm = surface_model(plane_grid, x, y, z, cls, dem)
    assert dsm[1, 1] == 4.0  # the high-noise return at 55 m is ignored
    assert dsm[2, 2] < 9.0  # the roof is not vegetation


def test_empty_cells_take_neighbor_mean_and_never_drop_below_ground(plane_grid: Grid) -> None:
    dem = np.full(plane_grid.shape, 2.0)
    x, y, z, cls = _points_at(plane_grid, [(10, 9), (10, 11)], [6.0, 8.0], [1, 1])
    dsm = surface_model(plane_grid, x, y, z, cls, dem)
    assert dsm[10, 10] == pytest.approx(7.0)
    assert dsm[25, 15] == 2.0
    assert (dsm >= dem).all()


def test_chm_clips_spikes_and_fills_pits() -> None:
    dem = np.zeros((7, 7))
    dsm = np.full((7, 7), 8.0)
    dsm[3, 3] = 0.5  # a pit inside a crown
    dsm[1, 5] = 80.0  # an unflagged spike
    dsm[6, :] = -1.0  # returns below ground
    chm = canopy_height_model(dsm, dem, CanopyParams(max_height_m=40.0, pit_fill_m=1.5))
    assert chm[3, 3] == pytest.approx(8.0)
    assert chm[1, 5] == pytest.approx(8.0)
    assert chm[6, :].min() >= 0.0
    no_pit_fill = canopy_height_model(dsm, dem, CanopyParams(pit_fill_m=0.0))
    assert no_pit_fill[3, 3] == pytest.approx(0.5)


def test_crown_heights_match_truth(scene: SyntheticScene, products: Products) -> None:
    plants = scene.plants
    size = scene.params.size_m
    errors = []
    for i in np.flatnonzero((plants.kind == TREE) | (plants.kind == SMALL_TREE)):
        row, col = int(size - plants.v[i]), int(plants.u[i])
        window = (slice(max(row - 1, 0), row + 2), slice(max(col - 1, 0), col + 2))
        errors.append(products.chm[window].max() - scene.chm_true[window].max())
    errors_m = np.abs(np.array(errors))
    assert np.median(errors_m) < 0.15
    assert np.percentile(errors_m, 90) < 0.5
