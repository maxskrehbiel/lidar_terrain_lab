"""Scores pipeline output against the known truth of a synthetic scene."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .classify import class_areas, classify_heights, woody_acres
from .pipeline import Products
from .synthetic import SMALL_TREE, TREE, UNCLASSIFIED, SyntheticScene

CHANNEL_SAMPLE_START = 0.40  # share of the scene width
CHANNEL_SAMPLE_END = 0.95
CHANNEL_SEARCH_M = 30.0  # half-height of the window searched around the true channel
ELEVATED_M = 0.5  # returns higher than this above ground must not be labeled ground
TALL_CROWN_M = 2.0  # DEM error is also reported under crowns taller than this


@dataclass(frozen=True)
class Tolerances:
    """Pass limits for each check.

    Attributes:
        ground_recall_min: Share of true ground returns labeled ground.
        ground_commission_max: Share of returns more than 0.5 m up labeled ground.
        dem_rmse_max_m: DEM root-mean-square error over all cells.
        dem_rmse_under_canopy_max_m: DEM error where crowns hid most ground returns.
        crown_top_error_max_m: Median error of the tallest canopy height per crown.
        class_area_error_max: Relative area error per woody class.
        woody_area_error_max: Relative error of total woody area.
        channel_median_offset_max_m: Median distance from drainage line to true channel.
        channel_p95_offset_max_m: 95th-percentile distance to the true channel.
        channel_coverage_min: Share of sampled cross-sections where a drainage line exists.
        outlet_share_min: Share of the scene draining through the main outlet.
    """

    ground_recall_min: float = 0.97
    ground_commission_max: float = 0.01
    dem_rmse_max_m: float = 0.10
    dem_rmse_under_canopy_max_m: float = 0.20
    crown_top_error_max_m: float = 0.5
    class_area_error_max: float = 0.15
    woody_area_error_max: float = 0.10
    channel_median_offset_max_m: float = 2.0
    channel_p95_offset_max_m: float = 5.0
    channel_coverage_min: float = 0.95
    outlet_share_min: float = 0.5


@dataclass(frozen=True)
class Check:
    """One validation metric.

    Attributes:
        key: Stable identifier, e.g. ``dem_rmse`` or ``area_error_shrub``.
        name: What was measured, for display.
        value: Measured value.
        limit: Pass limit.
        passed: Whether ``value`` is on the right side of ``limit``.
        unit: Unit or kind of value, for display.
        comparison: ``>=`` when larger is better, ``<=`` when smaller is better.
    """

    key: str
    name: str
    value: float
    limit: float
    passed: bool
    unit: str
    comparison: str


def _at_least(key: str, name: str, value: float, limit: float, unit: str) -> Check:
    return Check(key, name, value, limit, value >= limit, unit, ">=")


def _at_most(key: str, name: str, value: float, limit: float, unit: str) -> Check:
    return Check(key, name, value, limit, value <= limit, unit, "<=")


def _ground_checks(products: Products, scene: SyntheticScene, tol: Tolerances) -> list[Check]:
    truth = scene.is_ground
    elevated = (
        ~truth
        & (scene.height_above_ground > ELEVATED_M)
        & (scene.cloud.classification == UNCLASSIFIED)
    )
    recall = float(products.is_ground[truth].mean())
    commission = float(products.is_ground[elevated].mean())
    return [
        _at_least(
            "ground_recall", "Ground returns labeled ground", recall, tol.ground_recall_min, "share"
        ),
        _at_most(
            "ground_commission",
            "Vegetation returns labeled ground",
            commission,
            tol.ground_commission_max,
            "share",
        ),
    ]


def _dem_checks(products: Products, scene: SyntheticScene, tol: Tolerances) -> list[Check]:
    error = products.dem - scene.ground_true
    under = scene.chm_true > TALL_CROWN_M
    return [
        _at_most(
            "dem_rmse",
            "DEM error, all cells (RMSE)",
            float(np.sqrt(np.mean(error**2))),
            tol.dem_rmse_max_m,
            "m",
        ),
        _at_most(
            "dem_rmse_under_crowns",
            f"DEM error under crowns >{TALL_CROWN_M:g} m (RMSE)",
            float(np.sqrt(np.mean(error[under] ** 2))),
            tol.dem_rmse_under_canopy_max_m,
            "m",
        ),
    ]


def _crown_check(products: Products, scene: SyntheticScene, tol: Tolerances) -> Check:
    """Tallest canopy height near each small tree and tree, measured versus true."""
    plants = scene.plants
    res = products.grid.res_m
    size = scene.params.size_m
    errors = []
    for i in np.flatnonzero((plants.kind == SMALL_TREE) | (plants.kind == TREE)).tolist():
        col = int(plants.u[i] / res)
        row = int((size - plants.v[i]) / res)
        half = max(1, int(0.5 * plants.radius[i] / res))
        window = (
            slice(max(0, row - half), row + half + 1),
            slice(max(0, col - half), col + half + 1),
        )
        errors.append(abs(float(products.chm[window].max() - scene.chm_true[window].max())))
    return _at_most(
        "crown_top_error",
        "Crown-top height error (median, small trees and trees)",
        float(np.median(errors)),
        tol.crown_top_error_max_m,
        "m",
    )


def _area_checks(products: Products, scene: SyntheticScene, tol: Tolerances) -> list[Check]:
    truth_codes = classify_heights(scene.chm_true, products.classes)
    truth_areas = class_areas(truth_codes, products.classes, products.grid.cell_area_m2)
    checks = []
    for truth, found in zip(truth_areas, products.class_table, strict=True):
        if not truth.woody:
            continue
        checks.append(
            _at_most(
                f"area_error_{truth.name.lower().replace(' ', '_')}",
                f"{truth.name} area error ({truth.acres:.2f} ac true, {found.acres:.2f} found)",
                abs(found.acres - truth.acres) / truth.acres,
                tol.class_area_error_max,
                "relative",
            )
        )
    woody_truth = woody_acres(truth_areas)
    woody_found = woody_acres(products.class_table)
    checks.append(
        _at_most(
            "woody_area_error",
            f"Woody area error ({woody_truth:.2f} ac true, {woody_found:.2f} found)",
            abs(woody_found - woody_truth) / woody_truth,
            tol.woody_area_error_max,
            "relative",
        )
    )
    return checks


def _channel_offsets(products: Products, scene: SyntheticScene) -> tuple[list[float], float]:
    """Distance from the drainage line to the true channel at each sampled cross-section."""
    grid = products.grid
    res = grid.res_m
    size = scene.params.size_m
    centerline = cKDTree(scene.channel_uv)
    channel_v = np.interp(
        (np.arange(grid.width) + 0.5) * res, scene.channel_uv[:, 0], scene.channel_uv[:, 1]
    )
    offsets: list[float] = []
    # Upstream, the channel may not yet drain enough area to count as a drainage line (random
    # uplands can send much of the western catchment off the scene edges), so it is scored
    # only along its lower reach, where the catchment is certainly above the threshold.
    first, last = int(CHANNEL_SAMPLE_START * grid.width), int(CHANNEL_SAMPLE_END * grid.width)
    for col in range(first, last):
        v_true = channel_v[col]
        top = max(0, int((size - v_true - CHANNEL_SEARCH_M) / res))
        bottom = min(grid.height, int((size - v_true + CHANNEL_SEARCH_M) / res) + 1)
        rows = np.arange(top, bottom)
        on_channel = rows[products.hydrology.channel_mask[rows, col]]
        if not on_channel.size:
            continue
        # Nearest drainage cell, so a tributary crossing the window cannot stand in for it.
        row_true = (size - v_true) / res - 0.5
        best = on_channel[np.argmin(np.abs(on_channel - row_true))]
        distance, _ = centerline.query(((col + 0.5) * res, size - (best + 0.5) * res))
        offsets.append(float(distance))
    sampled = max(last - first, 1)
    return offsets, len(offsets) / sampled


def _channel_checks(products: Products, scene: SyntheticScene, tol: Tolerances) -> list[Check]:
    offsets, coverage = _channel_offsets(products, scene)
    median = float(np.median(offsets)) if offsets else float("inf")
    p95 = float(np.percentile(offsets, 95)) if offsets else float("inf")
    accumulation = products.hydrology.flow_accumulation
    outlet_share = float(accumulation[:, -1].max()) / accumulation.size
    return [
        _at_least(
            "channel_coverage",
            "Lower-channel cross-sections with a drainage line",
            coverage,
            tol.channel_coverage_min,
            "share",
        ),
        _at_most(
            "channel_offset_median",
            "Drainage line to true lower channel (median)",
            median,
            tol.channel_median_offset_max_m,
            "m",
        ),
        _at_most(
            "channel_offset_p95",
            "Drainage line to true lower channel (95th pct)",
            p95,
            tol.channel_p95_offset_max_m,
            "m",
        ),
        _at_least(
            "outlet_share",
            "Share of scene draining through the east outlet",
            outlet_share,
            tol.outlet_share_min,
            "share",
        ),
    ]


def validate(
    products: Products, scene: SyntheticScene, tol: Tolerances | None = None
) -> list[Check]:
    """Compare every product with the scene's truth.

    Args:
        products: Pipeline output for ``scene.cloud``.
        scene: The synthetic scene.
        tol: Pass limits (defaults to :class:`Tolerances`).

    Returns:
        One check per metric, in a fixed order.
    """
    tol = tol or Tolerances()
    return [
        *_ground_checks(products, scene, tol),
        *_dem_checks(products, scene, tol),
        _crown_check(products, scene, tol),
        *_area_checks(products, scene, tol),
        *_channel_checks(products, scene, tol),
    ]
