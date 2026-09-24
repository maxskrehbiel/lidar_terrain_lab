"""Shared fixtures: one small seeded synthetic scene and its pipeline output."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lidar_terrain_lab.config import PipelineParams
from lidar_terrain_lab.pipeline import Products, run_pipeline
from lidar_terrain_lab.rasterize import Grid
from lidar_terrain_lab.synthetic import SceneParams, SyntheticScene, make_scene

SCENE_SEED = 5
SCENE_SIZE_M = 160.0
CHANNEL_THRESHOLD_HA = 0.1


@pytest.fixture(scope="session")
def scene() -> SyntheticScene:
    """A 160 m synthetic scene (about 6 acres) with known truth."""
    return make_scene(SCENE_SEED, SceneParams(size_m=SCENE_SIZE_M))


@pytest.fixture(scope="session")
def params() -> PipelineParams:
    """Default settings with a drainage threshold sized for the small scene."""
    base = PipelineParams()
    return replace(
        base, hydrology=replace(base.hydrology, channel_threshold_ha=CHANNEL_THRESHOLD_HA)
    )


@pytest.fixture(scope="session")
def products(scene: SyntheticScene, params: PipelineParams) -> Products:
    """Pipeline output for the shared scene, with its three synthetic zones."""
    return run_pipeline(scene.cloud, params, scene.zones, grid=scene.grid)


@pytest.fixture
def plane_grid() -> Grid:
    """A 20 x 30 grid of 1 m cells in UTM zone 31N."""
    return Grid(x0=500_000.0, y1=1_030.0, res=1.0, width=20, height=30, crs="EPSG:32631")
