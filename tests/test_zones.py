"""Management zones: reading, rasterizing and per-zone acreage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rasterio.warp import transform as transform_points

from lidar_terrain_lab.classify import build_classes
from lidar_terrain_lab.config import VegetationParams
from lidar_terrain_lab.errors import InputDataError
from lidar_terrain_lab.pipeline import Products
from lidar_terrain_lab.rasterize import Grid
from lidar_terrain_lab.zones import (
    rasterize_zones,
    read_feature_collection,
    read_zones,
    summarize_zones,
    zones_from_geojson,
)


def _square_lonlat(grid: Grid, x0: float, y0: float, side: float) -> dict[str, Any]:
    xs = [x0, x0 + side, x0 + side, x0, x0]
    ys = [y0, y0, y0 + side, y0 + side, y0]
    lon, lat = transform_points(grid.crs, "EPSG:4326", xs, ys)
    return {"type": "Polygon", "coordinates": [[list(p) for p in zip(lon, lat, strict=True)]]}


def _collection(*geometries: dict[str, Any], names: tuple[str, ...] = ()) -> dict[str, Any]:
    features = []
    for i, geometry in enumerate(geometries):
        properties = {"name": names[i]} if i < len(names) else {}
        features.append({"type": "Feature", "properties": properties, "geometry": geometry})
    return {"type": "FeatureCollection", "features": features}


def test_square_zone_covers_the_right_cells(plane_grid: Grid) -> None:
    square = _square_lonlat(plane_grid, 500_002.0, 1_005.0, 10.0)
    zones = zones_from_geojson(_collection(square, names=("Pasture",)), plane_grid)
    burned = rasterize_zones(zones, plane_grid)
    assert zones[0].name == "Pasture"
    assert int((burned == 1).sum()) == 100


def test_unnamed_zones_and_non_polygons(plane_grid: Grid) -> None:
    point = {"type": "Point", "coordinates": [0.0, 0.0]}
    square = _square_lonlat(plane_grid, 500_001.0, 1_001.0, 4.0)
    zones = zones_from_geojson(_collection(point, square), plane_grid)
    assert [z.name for z in zones] == ["Zone 1"]


def test_legacy_crs_member_is_honored(plane_grid: Grid) -> None:
    ring = [[500_000.0, 1_000.0], [500_010.0, 1_000.0], [500_010.0, 1_010.0], [500_000.0, 1_000.0]]
    collection = _collection({"type": "Polygon", "coordinates": [ring]})
    collection["crs"] = {"type": "name", "properties": {"name": "EPSG:32631"}}
    zones = zones_from_geojson(collection, plane_grid)
    assert zones[0].geometry["coordinates"][0][1] == pytest.approx((500_010.0, 1_000.0))


def test_zone_errors(plane_grid: Grid) -> None:
    with pytest.raises(InputDataError, match="no Polygon"):
        zones_from_geojson(_collection(), plane_grid)
    unreferenced = Grid(0.0, 10.0, 1.0, 10, 10)
    with pytest.raises(InputDataError, match="georeferenced"):
        zones_from_geojson(_collection(), unreferenced)


@pytest.mark.parametrize(
    ("content", "message"),
    [(None, "cannot read"), ("{not json", "not valid JSON"), ("[1, 2]", "FeatureCollection")],
)
def test_bad_zone_files(tmp_path: Path, content: str | None, message: str) -> None:
    path = tmp_path / "zones.geojson"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(InputDataError, match=message):
        read_feature_collection(path)


def test_read_zones_from_file(plane_grid: Grid, tmp_path: Path) -> None:
    path = tmp_path / "zones.geojson"
    square = _square_lonlat(plane_grid, 500_000.0, 1_000.0, 5.0)
    path.write_text(json.dumps(_collection(square, names=("A",))), encoding="utf-8")
    assert read_zones(path, plane_grid)[0].name == "A"


def test_summaries_partition_the_scene(products: Products) -> None:
    assert [z.name for z in products.zone_table] == ["Unit A", "Unit B", "Unit C"]
    total = sum(z.total_acres for z in products.zone_table)
    assert total == pytest.approx(products.stats.area_acres, rel=0.005)
    woody = sum(z.woody_acres for z in products.zone_table)
    assert woody == pytest.approx(products.stats.woody_acres, rel=0.01)
    for zone in products.zone_table:
        assert sum(a.acres for a in zone.classes) == pytest.approx(zone.total_acres)
        assert 0.0 <= zone.mean_slope_deg < 30.0


def test_empty_zone_has_undefined_slope(plane_grid: Grid) -> None:
    square = _square_lonlat(plane_grid, 500_002.0, 1_005.0, 3.0)
    zones = zones_from_geojson(_collection(square), plane_grid)
    classes = build_classes(VegetationParams())
    summary = summarize_zones(
        np.zeros(plane_grid.shape, dtype=np.int32),
        zones,
        np.ones(plane_grid.shape, dtype=np.uint8),
        classes,
        np.zeros(plane_grid.shape),
        1.0,
    )
    assert summary[0].total_acres == 0.0
    assert np.isnan(summary[0].mean_slope_deg)
