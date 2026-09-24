"""Google Earth KMZ packaging."""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree

from lidar_terrain_lab.export import drainage_geojson, zones_geojson
from lidar_terrain_lab.kmz import ZIP_TIMESTAMP, write_kmz
from lidar_terrain_lab.pipeline import Products

KML = "{http://www.opengis.net/kml/2.2}"


def _write(products: Products, path: Path, with_zones: bool = True) -> Path:
    drainage = drainage_geojson(products.hydrology.segments, products.grid)
    zones = zones_geojson(products.zones, products.zone_table, products.grid)
    return write_kmz(
        path, products, "Demo & test", "synthetic", drainage, zones if with_zones else None
    )


def test_kmz_contains_overlays_lines_and_zones(products: Products, tmp_path: Path) -> None:
    with zipfile.ZipFile(_write(products, tmp_path / "out.kmz")) as archive:
        names = set(archive.namelist())
        root = ElementTree.fromstring(archive.read("doc.kml"))
        stamps = {info.date_time for info in archive.infolist()}
        overlay_png = archive.read("overlays/layer_1.png")
    assert "overlays/legend.png" in names
    overlays = root.findall(f".//{KML}GroundOverlay")
    assert len(overlays) == 5
    assert {f"overlays/layer_{i}.png" for i in range(5)} <= names
    north = overlays[0].find(f".//{KML}north")
    assert north is not None and 0.0 < float(north.text or "nan") < 0.02
    placemarks = root.findall(f".//{KML}Placemark")
    assert len(placemarks) == len(products.hydrology.segments) + len(products.zones)
    name = root.find(f"{KML}Document/{KML}name")
    assert name is not None and name.text == "Demo & test"
    assert stamps == {ZIP_TIMESTAMP}
    assert b"Software" not in overlay_png


def test_kmz_is_byte_identical_across_runs(products: Products, tmp_path: Path) -> None:
    first = _write(products, tmp_path / "a.kmz").read_bytes()
    second = _write(products, tmp_path / "b.kmz").read_bytes()
    assert first == second


def test_kmz_without_zones(products: Products, tmp_path: Path) -> None:
    path = _write(products, tmp_path / "plain.kmz", with_zones=False)
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("doc.kml"))
    assert len(root.findall(f".//{KML}Placemark")) == len(products.hydrology.segments)
