"""Live queries against the public USGS TNM Access API (run with ``pytest -m integration``)."""

from __future__ import annotations

import pytest

from lidar_terrain_lab.config import load_demo_region_config
from lidar_terrain_lab.fetch import coverage, search_tiles, select_tiles


@pytest.mark.integration
def test_demo_region_has_full_3dep_coverage() -> None:
    region = load_demo_region_config().region
    assert region is not None
    tiles = search_tiles(region.bbox_wgs84)
    chosen = select_tiles(tiles, region.bbox_wgs84)
    assert chosen
    assert coverage(chosen, region.bbox_wgs84) == 1.0
    assert all(t.url.startswith("https://") and t.filename.endswith(".laz") for t in chosen)
    assert len({t.project for t in chosen}) == 1
