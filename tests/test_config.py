"""TOML configuration loading, including the packaged demo region."""

from __future__ import annotations

from pathlib import Path

import pytest

from lidar_terrain_lab.config import (
    PipelineParams,
    load_config,
    load_demo_region_config,
    parse_config,
)
from lidar_terrain_lab.errors import ConfigError


def test_packaged_demo_region_loads() -> None:
    config = load_demo_region_config()
    assert config.region is not None
    west, south, east, north = config.region.bbox_wgs84
    assert west < east and south < north
    assert "public land" in config.region.land_status.lower()
    assert config.pipeline.ground.terrain_slope == 0.45
    assert config.pipeline.vegetation.treatment == ("Shrub", "Small tree")


def test_config_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text("[pipeline]\nresolution_m = 2.0\n", encoding="utf-8")
    config = load_config(path)
    assert config.region is None
    assert config.pipeline.resolution_m == 2.0


def test_missing_tables_take_defaults() -> None:
    config = parse_config("")
    assert config.region is None
    assert config.pipeline == PipelineParams()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[pipeline]\nresolution = 2\n", "unknown key"),
        ("[pipeline.ground]\nwindow = 3\n", r"pipeline\.ground"),
        ('[region]\nname = "x"\nland_status = "y"\nbbox_wgs84 = [10, 0, 5, 1]\n', "invalid bbox"),
        ('[region]\nname = "x"\nland_status = "y"\nbbox_wgs84 = [1, 2, 3]\n', "invalid bbox"),
        ('[region]\nname = "x"\n', r"\[region\]"),
        ("[pipeline\n", "not valid TOML"),
    ],
)
def test_bad_configs_are_rejected(text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(text)


def test_missing_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read config"):
        load_config(tmp_path / "missing.toml")
