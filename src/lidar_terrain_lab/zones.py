"""User-supplied management zones: reading GeoJSON, rasterizing, and per-zone acreage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.warp import transform_geom

from ._types import FloatArray, Int32Array, UInt8Array
from .classify import ClassArea, VegClass, class_areas, treatment_acres, woody_acres
from .errors import InputDataError
from .rasterize import SQUARE_METERS_PER_ACRE, Grid

GEOJSON_CRS = "EPSG:4326"  # RFC 7946 fixes GeoJSON to longitude/latitude
POLYGON_TYPES = ("Polygon", "MultiPolygon")


@dataclass(frozen=True)
class Zone:
    """A named polygon in the grid's CRS.

    Attributes:
        zone_id: Raster value (1-based).
        name: Display name.
        geometry: GeoJSON geometry in the grid CRS.
    """

    zone_id: int
    name: str
    geometry: dict[str, Any]


@dataclass(frozen=True)
class ZoneSummary:
    """Acreage and terrain statistics for one zone.

    Attributes:
        zone_id: Raster value.
        name: Display name.
        total_acres: Zone area inside the grid.
        classes: Area per vegetation class.
        woody_acres: Area of the woody classes.
        treatment_acres: Area of the treatment classes.
        mean_slope_deg: Mean slope inside the zone.
    """

    zone_id: int
    name: str
    total_acres: float
    classes: list[ClassArea]
    woody_acres: float
    treatment_acres: float
    mean_slope_deg: float


def _source_crs(collection: dict[str, Any]) -> str:
    # Pre-2016 GeoJSON could name a CRS; honor it, otherwise follow RFC 7946.
    legacy = collection.get("crs", {}).get("properties", {}).get("name")
    return str(legacy) if legacy else GEOJSON_CRS


def zones_from_geojson(
    collection: dict[str, Any], grid: Grid, name_field: str = "name"
) -> list[Zone]:
    """Convert a GeoJSON FeatureCollection of polygons into zones in the grid CRS.

    Args:
        collection: Parsed GeoJSON.
        grid: Target grid (must have a CRS).
        name_field: Property holding each zone's name.

    Returns:
        Zones numbered from 1 in file order.

    Raises:
        InputDataError: If the grid has no CRS or the collection has no polygon features.
    """
    if grid.crs is None:
        raise InputDataError("zones need a georeferenced grid")
    source = CRS.from_user_input(_source_crs(collection))
    target = CRS.from_user_input(grid.crs)
    zones: list[Zone] = []
    for feature in collection.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") not in POLYGON_TYPES:
            continue
        zone_id = len(zones) + 1
        name = str((feature.get("properties") or {}).get(name_field) or f"Zone {zone_id}")
        zones.append(Zone(zone_id, name, dict(transform_geom(source, target, geometry))))
    if not zones:
        raise InputDataError("no Polygon or MultiPolygon features found")
    return zones


def read_feature_collection(path: str | Path) -> dict[str, Any]:
    """Read a GeoJSON FeatureCollection from disk.

    Args:
        path: GeoJSON file.

    Returns:
        The parsed collection.

    Raises:
        InputDataError: If the file is missing, is not JSON, or is not a FeatureCollection.
    """
    try:
        collection = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputDataError(f"cannot read {path}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise InputDataError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(collection, dict) or collection.get("type") != "FeatureCollection":
        raise InputDataError(f"{path} is not a GeoJSON FeatureCollection")
    return collection


def read_zones(path: str | Path, grid: Grid, name_field: str = "name") -> list[Zone]:
    """Read zones from a GeoJSON file.

    Args:
        path: GeoJSON file of polygons.
        grid: Target grid.
        name_field: Property holding each zone's name.

    Returns:
        Zones in the grid CRS.
    """
    return zones_from_geojson(read_feature_collection(path), grid, name_field)


def rasterize_zones(zones: list[Zone], grid: Grid) -> Int32Array:
    """Burn zone ids into the grid; a cell belongs to a zone when its center is inside.

    Args:
        zones: Zones in the grid CRS.
        grid: Target grid.

    Returns:
        Zone id per cell, 0 outside every zone. Later zones win where polygons overlap.
    """
    shapes = [(zone.geometry, zone.zone_id) for zone in zones]
    burned = rasterize(
        shapes, out_shape=grid.shape, transform=grid.transform, fill=0, dtype="int32"
    )
    return np.asarray(burned, dtype=np.int32)


def summarize_zones(
    zone_raster: Int32Array,
    zones: list[Zone],
    codes: UInt8Array,
    classes: list[VegClass],
    slope_deg: FloatArray,
    cell_area_m2: float,
) -> list[ZoneSummary]:
    """Cross-tabulate vegetation classes and slope by zone.

    Args:
        zone_raster: Output of :func:`rasterize_zones`.
        zones: The zones.
        codes: Vegetation class raster.
        classes: Class definitions.
        slope_deg: Slope raster.
        cell_area_m2: Area of one cell.

    Returns:
        One summary per zone, in zone order.
    """
    summaries = []
    for zone in zones:
        inside = zone_raster == zone.zone_id
        areas = class_areas(codes, classes, cell_area_m2, inside)
        summaries.append(
            ZoneSummary(
                zone_id=zone.zone_id,
                name=zone.name,
                total_acres=float(inside.sum()) * cell_area_m2 / SQUARE_METERS_PER_ACRE,
                classes=areas,
                woody_acres=woody_acres(areas),
                treatment_acres=treatment_acres(areas),
                mean_slope_deg=float(slope_deg[inside].mean()) if inside.any() else float("nan"),
            )
        )
    return summaries
