"""Finds and downloads public USGS 3DEP LiDAR tiles through The National Map (TNM) Access API.

Only listing and download happen here; nothing is uploaded and no credentials are needed.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import numpy as np

from .errors import FetchError

logger = logging.getLogger(__name__)

TNM_PRODUCTS_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"
LPC_DATASET = "Lidar Point Cloud (LPC)"
USER_AGENT = "lidar_terrain_lab/0.1 (+https://github.com/maxskrehbiel/lidar_terrain_lab)"
ALLOWED_DOWNLOAD_HOSTS = ("usgs.gov", "prd-tnm.s3.amazonaws.com")
COVERAGE_SAMPLES = 11  # per side of the bounding box when testing tile coverage
FULL_COVERAGE = 0.999
TNM_HOST = "tnmaccess.nationalmap.gov"
DOWNLOAD_CHUNK_BYTES = 1 << 20

Bounds = tuple[float, float, float, float]


@dataclass(frozen=True)
class LidarTile:
    """One downloadable LAZ tile listed by TNM.

    Attributes:
        title: TNM product title.
        project: 3DEP project name (tiles from one project share a CRS and acquisition).
        url: Download URL.
        size_bytes: Reported file size.
        publication_date: ISO date string.
        bbox: ``(west, south, east, north)`` in degrees.
    """

    title: str
    project: str
    url: str
    size_bytes: int
    publication_date: str
    bbox: Bounds

    @property
    def filename(self) -> str:
        """Base name of the download URL."""
        return PurePosixPath(urlparse(self.url).path).name


def _project_from(url: str, title: str) -> str:
    parts = PurePosixPath(urlparse(url).path).parts
    if "Projects" in parts:
        index = parts.index("Projects")
        if index + 1 < len(parts):
            return parts[index + 1]
    return title.rsplit(" ", 1)[0]


def parse_products(payload: dict[str, Any]) -> list[LidarTile]:
    """Turn a TNM ``products`` response into tiles, skipping entries without a LAZ URL.

    Args:
        payload: Decoded JSON from the products endpoint.

    Returns:
        Tiles in response order.
    """
    tiles = []
    for item in payload.get("items", []):
        url = item.get("downloadURL") or (item.get("urls") or {}).get("LAZ")
        box = item.get("boundingBox") or {}
        if not url or not str(url).lower().endswith(".laz") or not box:
            continue
        tiles.append(
            LidarTile(
                title=str(item.get("title", "")),
                project=_project_from(str(url), str(item.get("title", ""))),
                url=str(url),
                size_bytes=int(item.get("sizeInBytes") or 0),
                publication_date=str(item.get("publicationDate") or ""),
                bbox=(
                    float(box["minX"]),
                    float(box["minY"]),
                    float(box["maxX"]),
                    float(box["maxY"]),
                ),
            )
        )
    return tiles


def search_tiles(bbox: Bounds, timeout_s: float = 60.0, max_items: int = 200) -> list[LidarTile]:
    """List 3DEP LAZ tiles that intersect a longitude/latitude box.

    Args:
        bbox: ``(west, south, east, north)`` in degrees.
        timeout_s: Network timeout.
        max_items: Page size requested from the API.

    Returns:
        All listed tiles (possibly from several projects).

    Raises:
        FetchError: On network or decoding failure, or a redirect away from the TNM host.
    """
    query = urlencode(
        {
            "bbox": ",".join(f"{v:.6f}" for v in bbox),
            "datasets": LPC_DATASET,
            "prodFormats": "LAZ",
            "max": str(max_items),
            "outputFormat": "JSON",
        }
    )
    request = Request(f"{TNM_PRODUCTS_URL}?{query}", headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout_s) as response:
            final = urlparse(response.geturl())
            if final.scheme != "https" or final.hostname != TNM_HOST:
                raise FetchError(f"TNM query was redirected to {response.geturl()}")
            payload = json.load(response)
    except (OSError, json.JSONDecodeError) as exc:
        raise FetchError(f"TNM products query failed: {exc}") from exc
    return parse_products(payload)


def coverage(tiles: Sequence[LidarTile], bbox: Bounds) -> float:
    """Share of an evenly spaced lattice of points in ``bbox`` that some tile covers.

    Args:
        tiles: Candidate tiles.
        bbox: ``(west, south, east, north)`` in degrees.

    Returns:
        A number on ``[0, 1]``.
    """
    lon = np.linspace(bbox[0], bbox[2], COVERAGE_SAMPLES)
    lat = np.linspace(bbox[1], bbox[3], COVERAGE_SAMPLES)
    grid_lon, grid_lat = np.meshgrid(lon, lat)
    covered = np.zeros(grid_lon.shape, dtype=bool)
    for tile in tiles:
        west, south, east, north = tile.bbox
        covered |= (
            (grid_lon >= west) & (grid_lon <= east) & (grid_lat >= south) & (grid_lat <= north)
        )
    return float(covered.mean())


def _intersects(a: Bounds, b: Bounds) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def select_tiles(tiles: Sequence[LidarTile], bbox: Bounds) -> list[LidarTile]:
    """Pick the tiles of one project that best cover the box.

    Prefers projects that cover the whole box, then the most recent publication, then
    the smallest total download.

    Args:
        tiles: Output of :func:`search_tiles`.
        bbox: ``(west, south, east, north)`` in degrees.

    Returns:
        Tiles to download, all from one project.

    Raises:
        FetchError: If no tile intersects the box.
    """
    by_project: dict[str, list[LidarTile]] = {}
    for tile in tiles:
        if _intersects(tile.bbox, bbox):
            by_project.setdefault(tile.project, []).append(tile)
    if not by_project:
        raise FetchError("no 3DEP LAZ tiles intersect the requested box")

    shares = {project: coverage(members, bbox) for project, members in by_project.items()}

    def rank(project: str) -> tuple[float, str, int]:
        members = by_project[project]
        # Any project that covers the whole box ranks above every partial one.
        full = 1.0 if shares[project] >= FULL_COVERAGE else shares[project]
        newest = max(t.publication_date for t in members)
        return (full, newest, -sum(t.size_bytes for t in members))

    best = max(by_project, key=rank)
    share = shares[best]
    if share < FULL_COVERAGE:
        logger.warning("project %s covers only %.0f%% of the box", best, 100 * share)
    return sorted(by_project[best], key=lambda t: t.filename)


def _check_url(url: str) -> None:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    allowed = any(host == h or host.endswith("." + h) for h in ALLOWED_DOWNLOAD_HOSTS)
    if parsed.scheme != "https" or not allowed:
        raise FetchError(f"refusing to download from unexpected location: {url}")


def download_tiles(
    tiles: Sequence[LidarTile],
    dest: str | Path,
    timeout_s: float = 120.0,
    opener: Callable[..., Any] = urlopen,
) -> list[Path]:
    """Download tiles into ``dest``, skipping files already present at the reported size.

    Args:
        tiles: Tiles from :func:`select_tiles`.
        dest: Destination directory (created if needed).
        timeout_s: Network timeout per request.
        opener: ``urlopen``-compatible callable (replaceable in tests).

    Returns:
        Local paths in tile order.

    Raises:
        FetchError: For a non-USGS URL (before or after redirects), an unsafe file name, or a
            failed download.
    """
    folder = Path(dest)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for tile in tiles:
        _check_url(tile.url)
        name = tile.filename
        if not name.lower().endswith(".laz") or name != Path(name).name:
            raise FetchError(f"unexpected file name in URL: {tile.url}")
        target = folder / name
        if target.exists() and tile.size_bytes and target.stat().st_size == tile.size_bytes:
            logger.info("already downloaded: %s", name)
            paths.append(target)
            continue
        partial = target.with_suffix(".laz.part")
        logger.info("downloading %s (%.1f MB)", name, tile.size_bytes / 1e6)
        request = Request(tile.url, headers={"User-Agent": USER_AGENT})
        try:
            with opener(request, timeout=timeout_s) as response:
                _check_url(response.geturl())  # a redirect must not leave the USGS hosts
                with partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle, DOWNLOAD_CHUNK_BYTES)
        except OSError as exc:
            partial.unlink(missing_ok=True)
            raise FetchError(f"download failed for {name}: {exc}") from exc
        partial.replace(target)
        paths.append(target)
    return paths
