"""Command-line interface, with network calls replaced by fakes."""

from __future__ import annotations

import argparse
import logging
import runpy
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from rasterio.warp import transform_bounds

from lidar_terrain_lab import cli, workflows
from lidar_terrain_lab.fetch import LidarTile
from lidar_terrain_lab.point_cloud import write_las
from lidar_terrain_lab.synthetic import SYNTHETIC_CRS, SyntheticScene
from lidar_terrain_lab.validation import Check

TILE = LidarTile(
    "USGS Lidar Point Cloud DEMO t1",
    "DEMO",
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/DEMO/LAZ/t1.laz",
    2_000_000_000,
    "2024-01-01",
    (-0.01, -0.01, 0.01, 0.01),
)


def _config(tmp_path: Path, bbox: tuple[float, float, float, float]) -> Path:
    path = tmp_path / "region.toml"
    path.write_text(
        '[region]\nname = "Test area"\nland_status = "synthetic"\n'
        f"bbox_wgs84 = [{', '.join(repr(v) for v in bbox)}]\n",
        encoding="utf-8",
    )
    return path


def _corner_laz(scene: SyntheticScene, tmp_path: Path) -> Path:
    corner = scene.cloud.subset(
        (scene.cloud.x < scene.cloud.x.min() + 40) & (scene.cloud.y < scene.cloud.y.min() + 40)
    )
    return write_las(corner, tmp_path / "corner.laz")


def test_demo_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["demo", "--out", str(tmp_path), "--size-m", "120", "--no-geotiff"])
    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert "Validation against synthetic truth" in out
    assert "[FAIL]" not in out
    assert not (tmp_path / "geotiff").exists()


def test_demo_exits_one_when_a_truth_check_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    failing = Check("dem_rmse", "DEM error", 9.9, 0.1, False, "m", "<=")
    monkeypatch.setattr(workflows, "validate", lambda products, scene: [failing])
    code = cli.main(["demo", "--out", str(tmp_path), "--size-m", "80", "--no-geotiff"])
    assert code == cli.EXIT_CHECKS_FAILED
    assert "[FAIL] DEM error" in capsys.readouterr().out


def test_demo_default_output_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.main(["demo", "--size-m", "80", "--no-geotiff"]) == cli.EXIT_OK
    assert (tmp_path / "demo_output" / "report.md").is_file()


@pytest.mark.parametrize("args", [["--size-m", "0"], ["--size-m", "abc"], ["--seed", "-1"]])
def test_invalid_demo_arguments_exit_two(args: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["demo", *args])
    assert exit_info.value.code == cli.EXIT_USAGE


def test_scene_too_small_is_a_clean_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["demo", "--out", str(tmp_path), "--size-m", "30"]) == cli.EXIT_USAGE
    assert capsys.readouterr().err.startswith("error: scene size")


def test_fetch_defaults_to_the_packaged_region(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    boxes: list[tuple[float, float, float, float]] = []
    regional = replace(TILE, bbox=(-99.0, 34.0, -98.0, 35.0))
    monkeypatch.setattr(cli, "search_tiles", lambda bbox: boxes.append(bbox) or [regional])
    assert cli.main(["fetch", "--list-only"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Wichita Mountains NWR sample area" in out and "t1.laz" in out
    assert boxes[0][0] < -98.0


def test_large_downloads_need_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    downloaded: list[Path] = []
    monkeypatch.setattr(cli, "search_tiles", lambda bbox: [TILE])
    monkeypatch.setattr(cli, "download_tiles", lambda tiles, dest: downloaded.append(dest) or [])
    config = _config(tmp_path, TILE.bbox)
    assert cli.main(["fetch", "--config", str(config)]) == cli.EXIT_OK
    assert "--yes" in capsys.readouterr().out and not downloaded
    assert cli.main(["fetch", "--config", str(config), "--yes"]) == cli.EXIT_OK
    assert downloaded
    assert cli.main(["run", "--config", str(config), "--fetch"]) == cli.EXIT_USAGE


def test_run_with_fetched_tiles(
    scene: SyntheticScene,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    laz = write_las(scene.cloud, tmp_path / "scene.laz")
    west, south, _, north = scene.grid.bounds
    box = transform_bounds(SYNTHETIC_CRS, "EPSG:4326", west, south, west + 60, north)
    small = LidarTile("t", "DEMO", TILE.url, 1000, "2024-01-01", box)
    monkeypatch.setattr(cli, "search_tiles", lambda bbox: [small])
    monkeypatch.setattr(cli, "download_tiles", lambda tiles, dest: [laz])
    out_dir = tmp_path / "out"
    config = _config(tmp_path, box)
    args = ["run", "--config", str(config), "--fetch", "--out", str(out_dir), "--no-geotiff"]
    assert cli.main(args) == cli.EXIT_OK
    assert (out_dir / "report.md").is_file()
    assert "ac processed" in capsys.readouterr().out


def test_run_with_local_tiles_needs_no_config(
    scene: SyntheticScene, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    laz = _corner_laz(scene, tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # no repository files are needed
    out_dir = tmp_path / "plain"
    args = ["run", "--laz", str(laz), "--ground", "pmf", "--out", str(out_dir), "--no-geotiff"]
    assert cli.main(args) == cli.EXIT_OK
    assert (out_dir / "report.md").is_file()


def test_run_default_output_folder(
    scene: SyntheticScene, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    laz = _corner_laz(scene, tmp_path)
    empty = tmp_path / "empty.toml"
    empty.write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["run", "--config", str(empty), "--laz", str(laz), "--no-geotiff"]) == 0
    assert (tmp_path / "outputs" / "run" / "report.md").is_file()


def test_input_and_config_errors_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty.toml"
    empty.write_text("", encoding="utf-8")
    assert cli.main(["fetch", "--config", str(empty)]) == cli.EXIT_USAGE
    missing = str(tmp_path / "missing.laz")
    assert cli.main(["run", "--laz", missing]) == cli.EXIT_USAGE
    assert cli.main(["run", "--config", str(tmp_path / "nope.toml"), "--laz", missing]) == 2
    err = capsys.readouterr().err
    assert err.count("error:") == 3 and "Traceback" not in err


def test_missing_optional_dependency_exits_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "laspy", None)
    code = cli.main(["run", "--laz", str(tmp_path / "any.laz")])
    assert code == cli.EXIT_MISSING_DEPENDENCY
    assert "lidar_terrain_lab[lidar]" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flags", "level"),
    [([], logging.WARNING), (["-v"], logging.INFO), (["-vv"], logging.DEBUG)],
)
def test_verbosity_levels(
    flags: list[str], level: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, int] = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: seen.update(kwargs))
    monkeypatch.setattr(cli, "search_tiles", lambda bbox: [TILE])
    cli.main([*flags, "fetch", "--config", str(_config(tmp_path, TILE.bbox)), "--list-only"])
    assert seen["level"] == level


def test_argument_types() -> None:
    assert cli.non_negative_int("0") == 0
    assert cli.positive_float("2.5") == 2.5
    with pytest.raises(argparse.ArgumentTypeError, match="whole number"):
        cli.non_negative_int("1.5")


def test_slug() -> None:
    assert cli._slug("Public Land Area, sample #1") == "public_land_area_sample_1"
    assert cli._slug("!!!") == "run"


def test_module_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["lidar_terrain_lab", "--version"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("lidar_terrain_lab", run_name="__main__")
    assert exit_info.value.code == 0
