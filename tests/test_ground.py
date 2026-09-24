"""Ground classification and DEM construction."""

from __future__ import annotations

import numpy as np
import pytest

from lidar_terrain_lab.config import GroundParams
from lidar_terrain_lab.ground import (
    bare_earth_dem,
    classify_ground,
    pmf_ground_mask,
    window_sizes,
)
from lidar_terrain_lab.point_cloud import PointCloud
from lidar_terrain_lab.rasterize import Grid
from lidar_terrain_lab.synthetic import SyntheticScene


def _sloped_ground_with_boxes(rng: np.random.Generator) -> tuple[PointCloud, np.ndarray]:
    """A 60 m square on a 10% slope with a 6 m "roof" and scattered 1.5 m shrubs."""
    n = 30_000
    x = rng.uniform(0, 60, n)
    y = rng.uniform(0, 60, n)
    ground = 50 + 0.1 * x
    z = ground + rng.normal(0, 0.02, n)
    roof = (x > 20) & (x < 32) & (y > 20) & (y < 30)
    shrub = ((x - 45) ** 2 + (y - 45) ** 2 < 4) & (rng.random(n) < 0.8)
    z = np.where(roof, ground + 6.0, z)
    z = np.where(shrub, ground + rng.uniform(0.6, 1.5, n), z)
    cloud = PointCloud(x, y, z, np.ones(n, dtype=np.uint8), crs="EPSG:32631")
    return cloud, ~(roof | shrub)


def test_window_sizes_double_until_the_limit() -> None:
    assert window_sizes(1.0, 20.0) == [3, 5, 9, 17]
    assert window_sizes(2.0, 20.0) == [3, 5, 9]
    assert window_sizes(1.0, 1.0) == [3]


def test_pmf_separates_ground_from_roof_and_shrubs() -> None:
    cloud, truth = _sloped_ground_with_boxes(np.random.default_rng(0))
    found = pmf_ground_mask(cloud.x, cloud.y, cloud.z, GroundParams())
    assert found[truth].mean() > 0.98
    assert found[~truth].mean() < 0.01


def test_auto_prefers_provider_ground_class() -> None:
    cloud, truth = _sloped_ground_with_boxes(np.random.default_rng(1))
    cloud.classification[truth] = 2
    result = classify_ground(cloud, GroundParams(method="auto"), cell_m=1.0)
    assert result.method == "existing"
    assert np.array_equal(result.is_ground, truth)


def test_auto_falls_back_to_the_filter_when_unclassified() -> None:
    cloud, _ = _sloped_ground_with_boxes(np.random.default_rng(2))
    assert classify_ground(cloud, GroundParams(method="auto"), cell_m=1.0).method == "pmf"


def test_existing_without_ground_class_is_an_error() -> None:
    cloud, _ = _sloped_ground_with_boxes(np.random.default_rng(3))
    with pytest.raises(ValueError, match="class 2"):
        classify_ground(cloud, GroundParams(method="existing"), cell_m=1.0)


def test_scene_ground_recovery(scene: SyntheticScene) -> None:
    keep = ~scene.cloud.noise_mask()
    cloud = scene.cloud.subset(keep)
    found = classify_ground(cloud, GroundParams(), cell_m=1.0).is_ground
    truth = scene.is_ground[keep]
    elevated = scene.height_above_ground[keep] > 0.5
    assert found[truth].mean() > 0.98
    assert found[elevated].mean() < 0.005


def test_bare_earth_dem_averages_and_fills(plane_grid: Grid) -> None:
    xs, ys = plane_grid.cell_centers()
    xx, yy = np.meshgrid(xs, ys)
    z = 10 + 0.2 * (xx - xx.min())
    keep = np.ones(xx.shape, dtype=bool)
    keep[5:9, 5:9] = False  # a gap, as under a dense crown
    dem, void = bare_earth_dem(plane_grid, xx[keep], yy[keep], z[keep])
    assert void == pytest.approx(16 / plane_grid.width / plane_grid.height)
    assert np.abs(dem - z).max() < 1e-3


def test_bare_earth_dem_without_points_is_an_error(plane_grid: Grid) -> None:
    empty = np.array([], dtype=np.float64)
    with pytest.raises(ValueError, match="no ground"):
        bare_earth_dem(plane_grid, empty, empty, empty)
