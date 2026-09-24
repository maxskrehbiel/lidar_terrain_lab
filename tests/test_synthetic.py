"""Seeded synthetic scene: determinism, truth consistency and parameter validation."""

from __future__ import annotations

import numpy as np
import pytest

from lidar_terrain_lab.errors import ConfigError
from lidar_terrain_lab.synthetic import (
    SYNTHETIC_CRS,
    SceneParams,
    SyntheticScene,
    SyntheticTerrain,
    _positions,
    block_max,
    crown_height,
    make_scene,
)


def test_same_seed_same_scene_and_different_seed_differs() -> None:
    first = make_scene(4, SceneParams(size_m=80.0))
    second = make_scene(4, SceneParams(size_m=80.0))
    other = make_scene(5, SceneParams(size_m=80.0))
    assert np.array_equal(first.cloud.z, second.cloud.z)
    assert len(first.cloud) != len(other.cloud) or not np.array_equal(first.cloud.z, other.cloud.z)


def test_ground_points_sit_on_the_true_surface(scene: SyntheticScene) -> None:
    ground = scene.is_ground
    assert scene.cloud.crs == SYNTHETIC_CRS
    assert np.abs(scene.height_above_ground[ground]).max() < 0.2
    assert (scene.cloud.classification[ground] == 1).all()
    noise = scene.cloud.noise_mask()
    assert noise.sum() == 2 * scene.params.noise_points
    assert not (ground & noise).any()


def test_truth_rasters_match_the_grid(scene: SyntheticScene) -> None:
    assert scene.ground_true.shape == scene.grid.shape == scene.chm_true.shape
    assert scene.chm_true.max() <= scene.plants.height.max() + 1e-9
    assert scene.chm_true.min() == 0.0


def test_zones_cover_the_scene_near_null_island(scene: SyntheticScene) -> None:
    features = scene.zones["features"]
    assert [f["properties"]["name"] for f in features] == ["Unit A", "Unit B", "Unit C"]
    coords = np.array([pt for f in features for pt in f["geometry"]["coordinates"][0]])
    assert coords.min() > 0.0 and coords.max() < 0.02


def test_crown_shape_and_block_max() -> None:
    du = np.array([0.0, 3.0, 5.0, 6.0])
    heights = crown_height(du, np.zeros(4), height=10.0, radius=5.0)
    assert heights[0] == 10.0 and heights[1] == pytest.approx(8.0) and heights[3] == 0.0
    fine = np.arange(16, dtype=float).reshape(4, 4)
    assert block_max(fine, 2).tolist() == [[5.0, 7.0], [13.0, 15.0]]


@pytest.mark.parametrize(
    "params",
    [
        SceneParams(size_m=40.0),
        SceneParams(size_m=100.5),
        SceneParams(res_m=0.0),
        SceneParams(trees_per_ha=-1.0),
        SceneParams(truth_supersample=0),
    ],
)
def test_invalid_scene_parameters(params: SceneParams) -> None:
    with pytest.raises(ConfigError):
        make_scene(1, params)


def test_placement_gives_up_instead_of_looping_forever() -> None:
    rng = np.random.default_rng(0)
    terrain = SyntheticTerrain(SceneParams(size_m=80.0), rng)
    with pytest.raises(ConfigError, match="no room"):
        _positions(rng, terrain, 5, min_channel_m=1e6, min_tributary_m=0.0)
