"""Demo and real-tile workflows, including maps and reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rasterio.warp import transform_bounds

from lidar_terrain_lab.config import PipelineParams, RegionConfig
from lidar_terrain_lab.point_cloud import write_las
from lidar_terrain_lab.synthetic import SYNTHETIC_CRS, SceneParams, SyntheticScene
from lidar_terrain_lab.workflows import run_demo, run_on_tiles

MAX_EXAMPLE_BYTES = 1_000_000


def test_demo_writes_every_output(tmp_path: Path) -> None:
    result = run_demo(tmp_path, seed=3, scene_params=SceneParams(size_m=140.0))
    out = result.outputs
    assert all(c.passed for c in result.checks)
    for path in [out.report_md, out.report_html, out.summary_json, out.kmz, out.drainage_geojson]:
        assert path.is_file() and 0 < path.stat().st_size < MAX_EXAMPLE_BYTES
    assert set(out.maps) == {
        "overview",
        "terrain",
        "canopy_height",
        "vegetation_classes",
        "drainage",
        "slope",
    }
    assert all(p.stat().st_size < MAX_EXAMPLE_BYTES for p in out.maps.values())
    assert {p.stem for p in out.geotiffs} >= {"dem", "chm"}
    assert (tmp_path / "vectors" / "zones_input.geojson").is_file()

    summary = json.loads(out.summary_json.read_text(encoding="utf-8"))
    assert summary["data"]["label"] == "SYNTHETIC"
    assert summary["grid"]["crs"] == SYNTHETIC_CRS
    assert len(summary["validation"]) == len(result.checks)
    report = out.report_md.read_text(encoding="utf-8")
    assert "Validation against synthetic truth" in report
    assert "maps/vegetation_classes.png" in report
    assert "<table>" in out.report_html.read_text(encoding="utf-8")


def test_run_on_tiles_with_region_crop_and_zones(scene: SyntheticScene, tmp_path: Path) -> None:
    laz = write_las(scene.cloud, tmp_path / "scene.laz")
    west, south, _, north = scene.grid.bounds
    box = transform_bounds(SYNTHETIC_CRS, "EPSG:4326", west, south, west + 80, north)
    region = RegionConfig("Test area", "synthetic", box)
    zones_path = tmp_path / "zones.geojson"
    zones_path.write_text(json.dumps(scene.zones), encoding="utf-8")
    products, outputs = run_on_tiles(
        [laz],
        tmp_path / "run",
        PipelineParams(),
        region=region,
        zones_path=zones_path,
        ground_method="pmf",
        write_geotiffs=False,
    )
    assert products.grid.width == pytest.approx(80, abs=2)
    assert products.stats.ground_method == "pmf"
    assert outputs.geotiffs == []
    assert outputs.zones_geojson is not None
    report = outputs.report_md.read_text(encoding="utf-8")
    assert "USGS 3DEP LiDAR" in report and "scene.laz" in report


def test_run_on_tiles_without_region(scene: SyntheticScene, tmp_path: Path) -> None:
    corner = scene.cloud.subset(
        (scene.cloud.x < scene.cloud.x.min() + 40) & (scene.cloud.y < scene.cloud.y.min() + 40)
    )
    laz = write_las(corner, tmp_path / "corner.laz")
    _, outputs = run_on_tiles([laz], tmp_path / "plain", PipelineParams(), write_geotiffs=False)
    report = outputs.report_md.read_text(encoding="utf-8")
    assert report.startswith("# LiDAR terrain report")
    assert "**Data: LiDAR.** Point cloud tiles: corner.laz." in report
    assert outputs.zones_geojson is None
