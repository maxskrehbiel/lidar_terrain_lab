"""Point cloud container and LAS/LAZ reading and writing (laspy is an optional dependency).

Coordinates stay in the file's projected CRS; heights are converted to meters.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from rasterio.crs import CRS
from rasterio.errors import CRSError
from rasterio.warp import transform_bounds

from ._types import BoolArray, FloatArray, UInt8Array
from .errors import InputDataError, LidarDependencyError

logger = logging.getLogger(__name__)

GROUND_CLASS = 2
NOISE_CLASSES = (7, 18)  # ASPRS low noise and high noise
US_SURVEY_FOOT_M = 1200.0 / 3937.0
INTERNATIONAL_FOOT_M = 0.3048

ZUnits = Literal["auto", "m", "ft", "us-ft"]
Bounds = tuple[float, float, float, float]


@dataclass
class PointCloud:
    """Columns of a classified point cloud.

    Attributes:
        x: Easting per point, CRS units.
        y: Northing per point, CRS units.
        z: Height per point, meters.
        classification: ASPRS class code per point (1 unclassified, 2 ground, 7/18 noise).
        crs: CRS as WKT or ``EPSG:`` string.
        meters_per_unit: Length of one horizontal CRS unit in meters.
        extent: Optional requested window ``(west, south, east, north)`` in CRS units.
        sources: File names the points were read from.
    """

    x: FloatArray
    y: FloatArray
    z: FloatArray
    classification: UInt8Array
    crs: str | None = None
    meters_per_unit: float = 1.0
    extent: Bounds | None = None
    sources: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        self.z = np.asarray(self.z, dtype=np.float64)
        self.classification = np.asarray(self.classification, dtype=np.uint8)
        n = len(self.x)
        if not (len(self.y) == len(self.z) == len(self.classification) == n):
            raise ValueError("x, y, z and classification must have the same length")

    def __len__(self) -> int:
        return len(self.x)

    def subset(self, mask: BoolArray) -> PointCloud:
        """Return the points where ``mask`` is true, keeping the metadata."""
        return PointCloud(
            self.x[mask],
            self.y[mask],
            self.z[mask],
            self.classification[mask],
            crs=self.crs,
            meters_per_unit=self.meters_per_unit,
            extent=self.extent,
            sources=list(self.sources),
        )

    @property
    def bounds(self) -> Bounds:
        """Tight ``(west, south, east, north)`` bounds of the points."""
        if len(self) == 0:
            raise ValueError("point cloud is empty")
        return (
            float(self.x.min()),
            float(self.y.min()),
            float(self.x.max()),
            float(self.y.max()),
        )

    def noise_mask(self) -> BoolArray:
        """True for points the data provider flagged as noise."""
        return np.isin(self.classification, NOISE_CLASSES)


def meters_per_crs_unit(crs: str) -> float:
    """Return the length in meters of one horizontal unit of a projected CRS.

    Args:
        crs: CRS as WKT or an ``EPSG:`` string.

    Returns:
        1.0 for meters, about 0.3048 for feet.

    Raises:
        InputDataError: If the CRS is not recognized or is geographic (degrees cannot be used
            for a raster in meters).
    """
    try:
        parsed = CRS.from_user_input(crs)
    except CRSError as exc:
        raise InputDataError(f"unrecognized CRS {crs!r}: {exc}") from exc
    if parsed.is_geographic:
        raise InputDataError("point clouds must be in a projected CRS, not latitude/longitude")
    try:
        _, factor = parsed.linear_units_factor
    except CRSError:
        logger.warning("could not read CRS linear units; assuming meters")
        return 1.0
    return float(factor)


def _import_laspy() -> Any:
    try:
        import laspy
    except ImportError as exc:
        raise LidarDependencyError(
            "reading LAS/LAZ needs the optional dependency: pip install 'lidar_terrain_lab[lidar]'"
        ) from exc
    return laspy


def las_crs(header: Any) -> str | None:
    """Extract the CRS from a LAS header without requiring pyproj.

    Tries laspy's own parser (which needs pyproj), then an OGC WKT VLR, then the
    GeoTIFF key directory's projected or geographic EPSG code.

    Args:
        header: A ``laspy.LasHeader``.

    Returns:
        WKT or an ``EPSG:`` string, or ``None`` if the header carries no CRS.
    """
    try:
        parsed = header.parse_crs()
    except (ImportError, RuntimeError, ValueError):
        # pyproj is optional and raises RuntimeError subclasses on malformed VLRs.
        parsed = None
    if parsed is not None:
        return str(parsed.to_wkt())
    vlrs = list(header.vlrs) + list(getattr(header, "evlrs", None) or [])
    for vlr in vlrs:
        kind = type(vlr).__name__
        if kind == "WktCoordinateSystemVlr":
            text = vlr.string
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="ignore")
            text = str(text).strip("\x00 \n")
            if text:
                return text
        if kind == "GeoKeyDirectoryVlr":
            for key in vlr.geo_keys:
                # 3072 = ProjectedCSTypeGeoKey, 2048 = GeographicTypeGeoKey; 32767 = user-defined
                if key.id in (3072, 2048) and key.value_offset not in (0, 32767):
                    return f"EPSG:{key.value_offset}"
    return None


def _z_factor(z_units: ZUnits, meters_per_unit: float) -> float:
    factors: dict[str, float] = {
        "m": 1.0,
        "ft": INTERNATIONAL_FOOT_M,
        "us-ft": US_SURVEY_FOOT_M,
    }
    # 3DEP deliveries use the same unit horizontally and vertically, so "auto" follows the CRS.
    return meters_per_unit if z_units == "auto" else factors[z_units]


def _tile_crs(laspy: Any, tile_paths: list[Path], crs: str | None) -> str:
    """The CRS shared by every tile (or the override), after checking they agree."""
    found: list[str] = []
    for path in tile_paths:
        if not path.is_file():
            raise InputDataError(f"file not found: {path}")
        try:
            with laspy.open(path) as reader:
                tile_crs = crs or las_crs(reader.header)
        except (laspy.errors.LaspyException, OSError) as exc:
            raise InputDataError(f"{path.name} is not a readable LAS/LAZ file: {exc}") from exc
        if tile_crs is None:
            raise InputDataError(f"{path.name} has no CRS in its header; pass --crs EPSG:xxxx")
        found.append(tile_crs)
    for path, tile_crs in zip(tile_paths[1:], found[1:], strict=True):
        if CRS.from_user_input(tile_crs) != CRS.from_user_input(found[0]):
            raise InputDataError(f"{path.name} uses a different CRS from the first tile")
    return found[0]


def _outside(window: Bounds, mins: Any, maxs: Any) -> bool:
    return bool(
        maxs[0] < window[0] or mins[0] > window[2] or maxs[1] < window[1] or mins[1] > window[3]
    )


def _read_tile(
    laspy: Any, path: Path, window: Bounds | None, chunk_size: int
) -> list[tuple[FloatArray, FloatArray, FloatArray, UInt8Array]]:
    """Points of one tile inside the window, skipping withheld points."""
    chunks = []
    with laspy.open(path) as reader:
        if window is not None and _outside(window, reader.header.mins, reader.header.maxs):
            logger.info("skipping %s: outside the requested window", path.name)
            return []
        for points in reader.chunk_iterator(chunk_size):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            keep = np.ones(len(x), dtype=bool)
            if window is not None:
                keep &= (x >= window[0]) & (x < window[2]) & (y >= window[1]) & (y < window[3])
            withheld = getattr(points, "withheld", None)
            if withheld is not None:
                keep &= ~np.asarray(withheld, dtype=bool)
            chunks.append(
                (
                    x[keep],
                    y[keep],
                    np.asarray(points.z, dtype=np.float64)[keep],
                    np.asarray(points.classification, dtype=np.uint8)[keep],
                )
            )
    logger.info("read %s", path.name)
    return chunks


def read_las(
    paths: Sequence[str | Path],
    *,
    bbox_wgs84: Bounds | None = None,
    crs: str | None = None,
    z_units: ZUnits = "auto",
    chunk_size: int = 2_000_000,
) -> PointCloud:
    """Read one or more LAS/LAZ tiles, optionally cropped to a longitude/latitude box.

    Args:
        paths: Tile paths; all must share one CRS.
        bbox_wgs84: Optional ``(west, south, east, north)`` in degrees to crop to.
        crs: Override for files whose header lacks a CRS.
        z_units: Vertical units of the file (``auto`` assumes they match the horizontal units).
        chunk_size: Points read per chunk, bounding peak memory.

    Returns:
        The cropped point cloud with heights in meters.

    Raises:
        LidarDependencyError: If laspy is not installed.
        InputDataError: If a file is missing or unreadable, no CRS can be determined, tiles
            disagree on CRS, or no points remain.
    """
    laspy = _import_laspy()
    tile_paths = [Path(p) for p in paths]
    if not tile_paths:
        raise InputDataError("no LAS/LAZ paths given")
    file_crs = _tile_crs(laspy, tile_paths, crs)
    unit = meters_per_crs_unit(file_crs)

    window: Bounds | None = None
    if bbox_wgs84 is not None:
        west, south, east, north = transform_bounds(
            CRS.from_epsg(4326), CRS.from_user_input(file_crs), *bbox_wgs84, densify_pts=21
        )
        window = (float(west), float(south), float(east), float(north))

    chunks = [c for path in tile_paths for c in _read_tile(laspy, path, window, chunk_size)]
    if sum(len(c[0]) for c in chunks) == 0:
        raise InputDataError("no points fall inside the requested window")
    return PointCloud(
        np.concatenate([c[0] for c in chunks]),
        np.concatenate([c[1] for c in chunks]),
        np.concatenate([c[2] for c in chunks]) * _z_factor(z_units, unit),
        np.concatenate([c[3] for c in chunks]),
        crs=file_crs,
        meters_per_unit=unit,
        extent=window,
        sources=[p.name for p in tile_paths],
    )


def write_las(cloud: PointCloud, path: str | Path) -> Path:
    """Write a point cloud to LAS 1.4 (``.laz`` compresses when lazrs is installed).

    Heights are written in meters, so this is intended for meter-based CRSs.

    Args:
        cloud: Points to write.
        path: Destination file.

    Returns:
        The written path.

    Raises:
        LidarDependencyError: If laspy is not installed.
    """
    laspy = _import_laspy()
    out = Path(path)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.floor([cloud.x.min(), cloud.y.min(), cloud.z.min()])
    header.scales = np.array([0.001, 0.001, 0.001])
    if cloud.crs is not None:
        wkt = CRS.from_user_input(cloud.crs).to_wkt()
        header.vlrs.append(laspy.vlrs.known.WktCoordinateSystemVlr(wkt))
        header.global_encoding.wkt = True
    data = laspy.LasData(header)
    data.x = cloud.x
    data.y = cloud.y
    data.z = cloud.z
    data.classification = cloud.classification
    data.write(out)
    return out
