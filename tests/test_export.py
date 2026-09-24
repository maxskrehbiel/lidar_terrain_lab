"""GeoTIFF and GeoJSON writers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lidar_terrain_lab.export import (
    drainage_geojson,
    raster_layers,
    read_geotiff,
    write_geojson,
    write_geotiff,
    write_raster_bundle,
    zones_geojson,
)
from lidar_terrain_lab.pipeline import Products
from lidar_terrain_lab.rasterize import Grid


def test_geotiff_round_trip_keeps_georeference_and_nodata(plane_grid: Grid, tmp_path: Path) -> None:
    data = np.arange(plane_grid.width * plane_grid.height, dtype=float).reshape(plane_grid.shape)
    data[0, 0] = np.nan
    path = write_geotiff(tmp_path / "a.tif", data, plane_grid, description="test band")
    back, grid = read_geotiff(path)
    assert np.isnan(back[0, 0])
    assert np.array_equal(back[1:], data[1:])
    assert (grid.x0, grid.y1, grid.res, grid.shape) == (
        plane_grid.x0,
        plane_grid.y1,
        plane_grid.res,
        plane_grid.shape,
    )
    assert grid.crs == "EPSG:32631"


def test_raster_bundle_writes_every_layer(products: Products, tmp_path: Path) -> None:
    paths = write_raster_bundle(products, tmp_path / "bundle")
    names = {p.stem for p in paths}
    assert {"dem", "chm", "flow_accumulation", "veg_class", "zones"} <= names
    assert len(paths) == len(raster_layers(products))
    chm, _ = read_geotiff(tmp_path / "bundle" / "chm.tif")
    assert np.nanmax(chm) == pytest.approx(products.chm.max(), abs=1e-4)


def test_drainage_geojson_is_longitude_latitude(products: Products, tmp_path: Path) -> None:
    segments = products.hydrology.segments
    collection = drainage_geojson(segments, products.grid)
    assert len(collection["features"]) == len(segments)
    lon, lat = collection["features"][0]["geometry"]["coordinates"][0]
    # The synthetic scene sits just north-east of 0 N, 0 E.
    assert 0.0 < lon < 0.02 and 0.0 < lat < 0.02
    path = write_geojson(tmp_path / "lines.geojson", collection)
    assert json.loads(path.read_text(encoding="utf-8")) == collection
    assert b"\r\n" not in path.read_bytes()


def test_zones_geojson_carries_acreage(products: Products) -> None:
    collection = zones_geojson(products.zones, products.zone_table, products.grid)
    props = [f["properties"] for f in collection["features"]]
    assert [p["name"] for p in props] == ["Unit A", "Unit B", "Unit C"]
    assert all(p["tree_acres"] >= 0 and "open_acres" in p for p in props)
