"""Vegetation-height classes and acreage."""

from __future__ import annotations

import numpy as np
import pytest

from lidar_terrain_lab.classify import (
    NODATA_CLASS,
    build_classes,
    class_areas,
    classify_heights,
    treatment_acres,
    woody_acres,
    woody_floor_m,
)
from lidar_terrain_lab.config import VegetationParams
from lidar_terrain_lab.errors import ConfigError
from lidar_terrain_lab.pipeline import Products
from lidar_terrain_lab.rasterize import SQUARE_METERS_PER_ACRE
from lidar_terrain_lab.synthetic import SyntheticScene

CLASSES = build_classes(VegetationParams())


def test_labels_and_flags() -> None:
    labels = [c.label for c in CLASSES]
    assert labels == ["Open (<0.5 m)", "Shrub (0.5-2 m)", "Small tree (2-5 m)", "Tree (>5 m)"]
    assert [c.woody for c in CLASSES] == [False, True, True, True]
    assert [c.treatment for c in CLASSES] == [False, True, True, False]
    assert woody_floor_m(CLASSES) == 0.5


def test_woody_flag_does_not_depend_on_class_order() -> None:
    params = VegetationParams(woody=("Tree",), treatment=("Tree",))
    classes = build_classes(params)
    assert woody_floor_m(classes) == 5.0
    assert woody_floor_m(build_classes(VegetationParams(woody=()))) == 0.0


@pytest.mark.parametrize(
    "params",
    [
        VegetationParams(breaks_m=(2.0, 0.5, 5.0)),
        VegetationParams(breaks_m=(0.0, 2.0, 5.0)),
        VegetationParams(names=("a", "b")),
        VegetationParams(treatment=("Cactus",)),
        VegetationParams(woody=("Vine",)),
    ],
)
def test_invalid_class_definitions(params: VegetationParams) -> None:
    with pytest.raises(ConfigError):
        build_classes(params)


def test_breaks_are_inclusive_below() -> None:
    heights = np.array([[0.0, 0.49, 0.5, 1.99], [2.0, 4.99, 5.0, np.nan]])
    assert classify_heights(heights, CLASSES).tolist() == [[1, 1, 2, 2], [3, 3, 4, NODATA_CLASS]]


def test_areas_percent_woody_and_treatment() -> None:
    codes = np.array([[1, 2, 2, 3], [4, 4, 4, NODATA_CLASS]], dtype=np.uint8)
    areas = class_areas(codes, CLASSES, cell_area_m2=SQUARE_METERS_PER_ACRE)
    assert [a.cells for a in areas] == [1, 2, 1, 3]
    assert [a.acres for a in areas] == [1.0, 2.0, 1.0, 3.0]
    assert sum(a.percent for a in areas) == pytest.approx(100.0)
    assert woody_acres(areas) == 6.0
    assert treatment_acres(areas) == 3.0
    masked = class_areas(codes, CLASSES, SQUARE_METERS_PER_ACRE, mask=codes == 4)
    assert [a.percent for a in masked] == [0.0, 0.0, 0.0, 100.0]
    empty = class_areas(codes, CLASSES, 1.0, mask=np.zeros(codes.shape, dtype=bool))
    assert all(a.percent == 0.0 for a in empty)


def test_scene_class_acreage_matches_truth(scene: SyntheticScene, products: Products) -> None:
    truth = class_areas(
        classify_heights(scene.chm_true, products.classes),
        products.classes,
        products.grid.cell_area_m2,
    )
    for true_area, found in zip(truth, products.class_table, strict=True):
        if true_area.woody:
            assert found.acres == pytest.approx(true_area.acres, rel=0.12), true_area.name
