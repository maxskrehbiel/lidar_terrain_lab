"""3DEP tile search, selection and download, all offline with canned responses."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from lidar_terrain_lab import fetch
from lidar_terrain_lab.errors import FetchError
from lidar_terrain_lab.fetch import (
    LidarTile,
    coverage,
    download_tiles,
    parse_products,
    search_tiles,
    select_tiles,
)

BOX = (-100.00, 35.00, -99.99, 35.01)
BASE = "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects"


class FakeResponse(io.BytesIO):
    """Stands in for an ``urlopen`` response, including the post-redirect URL."""

    def __init__(self, payload: bytes, url: str) -> None:
        super().__init__(payload)
        self.url = url

    def geturl(self) -> str:
        return self.url


def _item(
    project: str, name: str, box: tuple[float, float, float, float], date: str
) -> dict[str, Any]:
    return {
        "title": f"USGS Lidar Point Cloud {project} {name}",
        "downloadURL": f"{BASE}/{project}/LAZ/{name}.laz",
        "sizeInBytes": 1000,
        "publicationDate": date,
        "boundingBox": {"minX": box[0], "minY": box[1], "maxX": box[2], "maxY": box[3]},
    }


PAYLOAD = {
    "items": [
        # An older project whose single tile covers the whole box.
        _item("OLD_FULL", "tile_a", (-100.1, 34.9, -99.9, 35.1), "2015-01-01"),
        # A newer project that only covers the western half.
        _item("NEW_PARTIAL", "tile_b", (-100.1, 34.9, -99.995, 35.1), "2024-01-01"),
        # A product without a LAZ link is ignored.
        {"title": "metadata only", "downloadURL": f"{BASE}/X/meta.xml", "boundingBox": {}},
        # A tile far away.
        _item("ELSEWHERE", "tile_c", (-90.0, 30.0, -89.9, 30.1), "2025-01-01"),
    ]
}


def test_parse_products_keeps_only_laz_tiles() -> None:
    tiles = parse_products(PAYLOAD)
    assert [t.project for t in tiles] == ["OLD_FULL", "NEW_PARTIAL", "ELSEWHERE"]
    assert tiles[0].filename == "tile_a.laz"


def test_project_name_falls_back_to_title() -> None:
    item = _item("P", "t", BOX, "2020-01-01")
    item["downloadURL"] = "https://rockyweb.usgs.gov/other/t.laz"
    assert parse_products({"items": [item]})[0].project == "USGS Lidar Point Cloud P"


def test_coverage_and_selection() -> None:
    tiles = parse_products(PAYLOAD)
    assert coverage(tiles[:1], BOX) == 1.0
    assert 0.3 < coverage(tiles[1:2], BOX) < 0.7
    # Full coverage beats recency.
    assert [t.project for t in select_tiles(tiles, BOX)] == ["OLD_FULL"]
    # Among partial options the best coverage wins, with a warning.
    assert [t.project for t in select_tiles(tiles[1:], BOX)] == ["NEW_PARTIAL"]
    with pytest.raises(FetchError, match="no 3DEP"):
        select_tiles(tiles[2:], BOX)


def test_search_uses_the_products_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def fake_urlopen(request: Any, timeout: float) -> FakeResponse:
        seen["url"] = request.full_url
        return FakeResponse(json.dumps(PAYLOAD).encode(), request.full_url)

    monkeypatch.setattr(fetch, "urlopen", fake_urlopen)
    tiles = search_tiles(BOX)
    assert len(tiles) == 3
    assert seen["url"].startswith(fetch.TNM_PRODUCTS_URL)
    assert "Lidar+Point+Cloud" in seen["url"]


def test_search_refuses_a_redirect_off_the_tnm_host(monkeypatch: pytest.MonkeyPatch) -> None:
    def redirected(request: Any, timeout: float) -> FakeResponse:
        return FakeResponse(json.dumps(PAYLOAD).encode(), "https://example.com/products")

    monkeypatch.setattr(fetch, "urlopen", redirected)
    with pytest.raises(FetchError, match="redirected"):
        search_tiles(BOX)


def test_search_failure_is_a_fetch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(request: Any, timeout: float) -> FakeResponse:
        raise OSError("offline")

    monkeypatch.setattr(fetch, "urlopen", failing)
    with pytest.raises(FetchError, match="offline"):
        search_tiles(BOX)


def _tile(url: str, size: int = 5) -> LidarTile:
    return LidarTile("t", "P", url, size, "2020-01-01", BOX)


def test_download_writes_and_then_skips(tmp_path: Path) -> None:
    calls: list[str] = []

    def opener(request: Any, timeout: float) -> FakeResponse:
        calls.append(request.full_url)
        return FakeResponse(b"12345", request.full_url)

    tile = _tile(f"{BASE}/P/LAZ/demo.laz")
    first = download_tiles([tile], tmp_path, opener=opener)
    assert first[0].read_bytes() == b"12345"
    download_tiles([tile], tmp_path, opener=opener)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://rockyweb.usgs.gov/x/demo.laz",
        "https://example.com/demo.laz",
        f"{BASE}/P/LAZ/readme.txt",
    ],
)
def test_download_refuses_unexpected_urls(url: str, tmp_path: Path) -> None:
    with pytest.raises(FetchError):
        download_tiles([_tile(url)], tmp_path, opener=lambda *a, **k: FakeResponse(b"", url))


def test_download_refuses_a_redirect_off_usgs_hosts(tmp_path: Path) -> None:
    def redirected(request: Any, timeout: float) -> FakeResponse:
        return FakeResponse(b"payload", "https://example.com/demo.laz")

    with pytest.raises(FetchError, match="unexpected location"):
        download_tiles([_tile(f"{BASE}/P/LAZ/demo.laz")], tmp_path, opener=redirected)
    assert not list(tmp_path.iterdir())


def test_failed_download_leaves_no_partial_file(tmp_path: Path) -> None:
    def broken(request: Any, timeout: float) -> FakeResponse:
        raise OSError("connection reset")

    with pytest.raises(FetchError, match="connection reset"):
        download_tiles([_tile(f"{BASE}/P/LAZ/demo.laz")], tmp_path, opener=broken)
    assert not list(tmp_path.iterdir())
