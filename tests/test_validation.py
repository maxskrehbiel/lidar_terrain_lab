"""The pipeline recovers the synthetic truth, and the checks can fail."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lidar_terrain_lab.config import PipelineParams
from lidar_terrain_lab.pipeline import Products, run_pipeline
from lidar_terrain_lab.synthetic import SceneParams, SyntheticScene, make_scene
from lidar_terrain_lab.validation import Tolerances, validate

EXPECTED_KEYS = [
    "ground_recall",
    "ground_commission",
    "dem_rmse",
    "dem_rmse_under_crowns",
    "crown_top_error",
    "area_error_shrub",
    "area_error_small_tree",
    "area_error_tree",
    "woody_area_error",
    "channel_coverage",
    "channel_offset_median",
    "channel_offset_p95",
    "outlet_share",
]


def test_every_truth_check_passes(products: Products, scene: SyntheticScene) -> None:
    checks = validate(products, scene)
    assert [c.key for c in checks] == EXPECTED_KEYS
    failed = [f"{c.key}: {c.value:.3f} vs {c.limit}" for c in checks if not c.passed]
    assert not failed, failed


def test_recovered_values(products: Products, scene: SyntheticScene) -> None:
    by_key = {c.key: c for c in validate(products, scene)}
    assert by_key["ground_recall"].value > 0.98
    assert by_key["dem_rmse"].value < 0.06
    assert by_key["area_error_tree"].value < 0.12
    assert by_key["channel_offset_median"].value < 1.5
    assert by_key["channel_coverage"].comparison == ">="
    assert by_key["dem_rmse"].comparison == "<="


def test_impossible_tolerances_fail(products: Products, scene: SyntheticScene) -> None:
    strict = Tolerances(dem_rmse_max_m=0.0, ground_recall_min=1.01)
    failed = {c.key for c in validate(products, scene, strict) if not c.passed}
    assert {"dem_rmse", "ground_recall"} <= failed


@pytest.mark.parametrize("seed", [1, 2])
def test_other_seeds_also_pass(seed: int) -> None:
    scene = make_scene(seed, SceneParams(size_m=120.0))
    base = PipelineParams()
    params = replace(base, hydrology=replace(base.hydrology, channel_threshold_ha=0.05))
    checks = validate(run_pipeline(scene.cloud, params, grid=scene.grid), scene)
    assert all(c.passed for c in checks), [c.key for c in checks if not c.passed]
