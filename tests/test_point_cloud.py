"""Point cloud container and LAS/LAZ reading and writing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from rasterio.warp import transform_bounds

from lidar_terrain_lab.errors import InputDataError, LidarDependencyError
from lidar_terrain_lab.point_cloud import (
    US_SURVEY_FOOT_M,
    PointCloud,
    las_crs,
    meters_per_crs_unit,
    read_las,
    write_las,
)
from lidar_terrain_lab.synthetic import SYNTHETIC_CRS, SyntheticScene

pytest.importorskip("laspy")


def _cloud(n: int = 400, crs: str | None = SYNTHETIC_CRS) -> PointCloud:
    rng = np.random.default_rng(0)
    x = 166_600 + rng.uniform(0, 50, n)
    y = 1_000 + rng.uniform(0, 50, n)
    z = 300 + rng.uniform(0, 5, n)
    classification = rng.choice(np.array([1, 2, 7], dtype=np.uint8), n)
    return PointCloud(x, y, z, classification, crs=crs)


def test_container_basics() -> None:
    cloud = _cloud()
    assert len(cloud) == 400
    assert cloud.bounds[0] >= 166_600
    subset = cloud.subset(cloud.classification == 2)
    assert (subset.classification == 2).all()
    assert subset.crs == SYNTHETIC_CRS
    assert cloud.noise_mask().sum() == (cloud.classification == 7).sum()
    with pytest.raises(ValueError, match="same length"):
        PointCloud(np.zeros(2), np.zeros(2), np.zeros(3), np.zeros(2, dtype=np.uint8))
    with pytest.raises(ValueError, match="empty"):
        _ = cloud.subset(np.zeros(len(cloud), dtype=bool)).bounds


def test_meters_per_crs_unit() -> None:
    assert meters_per_crs_unit("EPSG:32614") == 1.0
    # NAD83 / Texas North Central (US survey feet)
    assert meters_per_crs_unit("EPSG:2276") == pytest.approx(US_SURVEY_FOOT_M)
    with pytest.raises(InputDataError, match="projected"):
        meters_per_crs_unit("EPSG:4326")
    with pytest.raises(InputDataError, match="unrecognized"):
        meters_per_crs_unit("EPSG:0")


def test_las_round_trip_with_crop(tmp_path: Path) -> None:
    cloud = _cloud()
    path = write_las(cloud, tmp_path / "tile.laz")
    lon_lat = transform_bounds(SYNTHETIC_CRS, "EPSG:4326", 166_600, 1_000, 166_625, 1_050)
    cropped = read_las([path], bbox_wgs84=lon_lat)
    inside = cloud.x < 166_625
    assert len(cropped) == pytest.approx(int(inside.sum()), abs=3)
    assert cropped.meters_per_unit == 1.0
    assert cropped.extent is not None
    assert cropped.sources == ["tile.laz"]
    full = read_las([path])
    assert np.allclose(np.sort(full.z), np.sort(cloud.z), atol=1e-3)


def test_z_units_override(tmp_path: Path) -> None:
    path = write_las(_cloud(), tmp_path / "tile.laz")
    metres = read_las([path])
    feet = read_las([path], z_units="ft")
    assert np.allclose(feet.z, metres.z * 0.3048)


def test_read_errors(tmp_path: Path, scene: SyntheticScene) -> None:
    with pytest.raises(InputDataError, match="no LAS"):
        read_las([])
    with pytest.raises(InputDataError, match="not found"):
        read_las([tmp_path / "missing.laz"])
    corrupt = tmp_path / "corrupt.laz"
    corrupt.write_bytes(b"not a point cloud")
    with pytest.raises(InputDataError, match="not a readable"):
        read_las([corrupt])
    no_crs = write_las(_cloud(crs=None), tmp_path / "no_crs.laz")
    with pytest.raises(InputDataError, match="no CRS"):
        read_las([no_crs])
    assert len(read_las([no_crs], crs=SYNTHETIC_CRS)) == 400
    other = _cloud(crs="EPSG:32632")
    other_path = write_las(other, tmp_path / "other.laz")
    first = write_las(_cloud(), tmp_path / "first.laz")
    with pytest.raises(InputDataError, match="different CRS"):
        read_las([first, other_path])
    far_box = (10.0, 10.0, 10.1, 10.1)
    with pytest.raises(InputDataError, match="no points"):
        read_las([first], bbox_wgs84=far_box)
    assert scene.cloud.crs == SYNTHETIC_CRS


class WktCoordinateSystemVlr:
    """Stand-in with the class name laspy uses for OGC WKT records."""

    def __init__(self, text: bytes) -> None:
        self.string = text


class GeoKey:
    def __init__(self, key_id: int, value: int) -> None:
        self.id = key_id
        self.value_offset = value


class GeoKeyDirectoryVlr:
    """Stand-in with the class name laspy uses for GeoTIFF key directories."""

    def __init__(self, keys: list[GeoKey]) -> None:
        self.geo_keys = keys


class _Header:
    def __init__(self, vlrs: list[object]) -> None:
        self.vlrs = vlrs

    def parse_crs(self) -> None:
        raise ImportError("pyproj not installed")


def test_las_crs_fallbacks_without_pyproj() -> None:
    wkt = _Header([WktCoordinateSystemVlr(b'PROJCS["demo"]\x00')])
    assert las_crs(wkt) == 'PROJCS["demo"]'
    geokeys = _Header([GeoKeyDirectoryVlr([GeoKey(1024, 1), GeoKey(3072, 26914)])])
    assert las_crs(geokeys) == "EPSG:26914"
    user_defined = _Header([GeoKeyDirectoryVlr([GeoKey(3072, 32767)])])
    assert las_crs(user_defined) is None


def test_missing_laspy_gives_a_clear_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "laspy", None)
    with pytest.raises(LidarDependencyError, match=r"\[lidar\]"):
        read_las([tmp_path / "any.laz"])
    with pytest.raises(LidarDependencyError):
        write_las(_cloud(), tmp_path / "any.laz")
