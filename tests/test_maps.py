"""Map plates: files, determinism and class coloring."""

from __future__ import annotations

from pathlib import Path

import pytest
from matplotlib.colors import to_rgba

from lidar_terrain_lab.maps import _class_colors, _nice_length, write_maps
from lidar_terrain_lab.pipeline import Products

PLATES = {"overview", "terrain", "canopy_height", "vegetation_classes", "drainage", "slope"}


def test_every_plate_is_written_small_and_untagged(products: Products, tmp_path: Path) -> None:
    maps = write_maps(products, tmp_path / "maps", "Test", "SYNTHETIC")
    assert set(maps) == PLATES
    for path in maps.values():
        data = path.read_bytes()
        assert data.startswith(b"\x89PNG")
        assert 0 < len(data) < 1_000_000
        assert b"Software" not in data


def test_plates_are_byte_identical_across_runs(products: Products, tmp_path: Path) -> None:
    first = write_maps(products, tmp_path / "a", "Test", "footer")
    second = write_maps(products, tmp_path / "b", "Test", "footer")
    for name in PLATES:
        assert first[name].read_bytes() == second[name].read_bytes(), name


def test_non_woody_classes_are_transparent(products: Products) -> None:
    cmap, norm = _class_colors(products.classes)
    for veg in products.classes:
        rgba = cmap(norm(veg.code))
        if veg.woody:
            assert rgba == pytest.approx(to_rgba(veg.color))
        else:
            assert rgba[3] == 0.0


@pytest.mark.parametrize(("span", "bar"), [(400.0, 100.0), (160.0, 20.0), (1000.0, 200.0)])
def test_scale_bar_uses_round_lengths(span: float, bar: float) -> None:
    assert _nice_length(span) == bar
