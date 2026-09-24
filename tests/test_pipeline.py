"""End-to-end pipeline behavior on the shared synthetic scene."""

from __future__ import annotations

import numpy as np
import pytest

from lidar_terrain_lab.config import PipelineParams
from lidar_terrain_lab.errors import InputDataError
from lidar_terrain_lab.pipeline import Products, grid_for, run_pipeline
from lidar_terrain_lab.point_cloud import PointCloud
from lidar_terrain_lab.synthetic import SyntheticScene


def test_products_share_one_grid(products: Products) -> None:
    shape = products.grid.shape
    rasters = (products.dem, products.chm, products.slope, products.hydrology.flow_accumulation)
    for raster in rasters:
        assert raster.shape == shape
    assert products.veg_codes.shape == shape
    assert products.zone_raster is not None and products.zone_raster.shape == shape


def test_statistics_are_consistent(products: Products, scene: SyntheticScene) -> None:
    stats = products.stats
    assert stats.ground_method == "pmf"
    assert stats.points_noise == 2 * scene.params.noise_points
    assert stats.points_total == len(scene.cloud)
    assert stats.area_hectares == pytest.approx((scene.params.size_m / 100) ** 2)
    assert sum(stats.slope_share_percent.values()) == pytest.approx(100.0)
    assert stats.treatment_classes == ("Shrub", "Small tree")
    assert stats.treatment_acres <= stats.woody_acres
    assert stats.canopy_max_m <= 12.0 + 0.1
    assert stats.channel_length_m > 0.5 * scene.params.size_m
    assert stats.channel_reaches == len(products.hydrology.segments)


def test_noise_is_never_ground(products: Products, scene: SyntheticScene) -> None:
    assert not products.is_ground[scene.cloud.noise_mask()].any()


def test_filled_dem_is_never_below_the_dem(products: Products) -> None:
    hydro = products.hydrology
    assert (hydro.filled_dem >= products.dem).all()
    assert np.array_equal(hydro.sink_depth > 0, hydro.filled_dem > products.dem)


def test_grid_for_uses_extent_or_point_bounds(scene: SyntheticScene) -> None:
    assert grid_for(scene.cloud, 2.0).shape == (80, 80)
    loose = PointCloud(
        scene.cloud.x[:1000],
        scene.cloud.y[:1000],
        scene.cloud.z[:1000],
        scene.cloud.classification[:1000],
        crs=scene.cloud.crs,
    )
    west, south, _, _ = loose.bounds
    grid = grid_for(loose, 1.0)
    assert grid.x0 <= west and grid.bounds[1] <= south


def test_pipeline_input_errors(scene: SyntheticScene) -> None:
    n = 100
    no_crs = PointCloud(
        scene.cloud.x[:n], scene.cloud.y[:n], scene.cloud.z[:n], scene.cloud.classification[:n]
    )
    with pytest.raises(InputDataError, match="CRS"):
        run_pipeline(no_crs, PipelineParams())
    all_noise = PointCloud(
        scene.cloud.x[:n],
        scene.cloud.y[:n],
        scene.cloud.z[:n],
        np.full(n, 7, dtype=np.uint8),
        crs=scene.cloud.crs,
    )
    with pytest.raises(InputDataError, match="noise"):
        run_pipeline(all_noise, PipelineParams())


def test_runs_without_zones(scene: SyntheticScene) -> None:
    small = scene.cloud.subset(scene.cloud.x < scene.cloud.x.min() + 60)
    products = run_pipeline(small, PipelineParams(resolution_m=2.0))
    assert products.zones == [] and products.zone_raster is None
    assert products.grid.res_m == 2.0
