"""Writes products to GIS formats: a GeoTIFF bundle and GeoJSON vectors."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform as transform_points
from rasterio.warp import transform_geom

from ._types import FloatArray
from .classify import NODATA_CLASS
from .hydrology import DrainageSegment, simplify_staircase
from .pipeline import Products
from .rasterize import Grid
from .zones import Zone, ZoneSummary

COORDINATE_DECIMALS = 7  # about 1 cm of longitude/latitude
FLOAT_NODATA = -9999.0


@dataclass(frozen=True)
class RasterLayer:
    """One GeoTIFF in the bundle.

    Attributes:
        name: File stem.
        data: Array on the product grid.
        dtype: GeoTIFF data type.
        nodata: Nodata value, or ``None``.
        description: Band description.
    """

    name: str
    data: np.ndarray[Any, np.dtype[Any]]
    dtype: str
    nodata: float | None
    description: str


def write_geotiff(
    path: str | Path,
    data: np.ndarray[Any, np.dtype[Any]],
    grid: Grid,
    dtype: str = "float32",
    nodata: float | None = FLOAT_NODATA,
    description: str = "",
) -> Path:
    """Write one band as a tiled, LZW-compressed GeoTIFF.

    Args:
        path: Output file.
        data: Array with the grid's shape (NaN becomes nodata for float types).
        grid: Grid giving CRS and transform.
        dtype: Output data type.
        nodata: Nodata value, or ``None``.
        description: Band description stored in the file.

    Returns:
        The written path.
    """
    out = Path(path)
    values = data
    if np.issubdtype(np.dtype(dtype), np.floating) and nodata is not None:
        values = np.where(np.isfinite(data), data, nodata)
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": 1,
        "dtype": dtype,
        "crs": CRS.from_user_input(grid.crs) if grid.crs else None,
        "transform": grid.transform,
        "nodata": nodata,
        "compress": "lzw",
    }
    if grid.width >= 256 and grid.height >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(values.astype(dtype), 1)
        if description:
            dst.set_band_description(1, description)
    return out


def read_geotiff(path: str | Path) -> tuple[FloatArray, Grid]:
    """Read band 1 of a GeoTIFF as float64 with nodata as NaN.

    Args:
        path: GeoTIFF file.

    Returns:
        The array and a grid describing it.
    """
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        crs = src.crs.to_string() if src.crs else None
        grid = Grid(src.transform.c, src.transform.f, src.transform.a, src.width, src.height, crs)
    return data, grid


def raster_layers(products: Products) -> list[RasterLayer]:
    """The GeoTIFF bundle's layers.

    Args:
        products: Pipeline output.

    Returns:
        Layers in a stable order.
    """
    hydro = products.hydrology
    layers = [
        RasterLayer("dem", products.dem, "float32", FLOAT_NODATA, "bare-earth elevation (m)"),
        RasterLayer("dsm", products.dsm, "float32", FLOAT_NODATA, "surface elevation (m)"),
        RasterLayer("chm", products.chm, "float32", FLOAT_NODATA, "canopy height (m)"),
        RasterLayer("slope", products.slope, "float32", FLOAT_NODATA, "slope (degrees)"),
        RasterLayer("aspect", products.aspect, "float32", FLOAT_NODATA, "aspect (deg from N)"),
        RasterLayer(
            "hillshade",
            np.round(products.hillshade * 254 + 1),
            "uint8",
            0,
            "shaded relief (1-255)",
        ),
        RasterLayer("sink_depth", hydro.sink_depth, "float32", FLOAT_NODATA, "fill depth (m)"),
        RasterLayer("flow_direction", hydro.flow_direction, "uint8", 255, "D8 code"),
        RasterLayer(
            "flow_accumulation",
            hydro.flow_accumulation,
            "float32",
            FLOAT_NODATA,
            "upstream cells",
        ),
        RasterLayer(
            "stream_order", hydro.stream_order, "uint8", 0, "Strahler order of drainage lines"
        ),
        RasterLayer("veg_class", products.veg_codes, "uint8", NODATA_CLASS, "vegetation class"),
    ]
    if products.zone_raster is not None:
        layers.append(RasterLayer("zones", products.zone_raster, "int16", 0, "zone id"))
    return layers


def write_raster_bundle(products: Products, out_dir: str | Path) -> list[Path]:
    """Write every raster layer as ``<out_dir>/<name>.tif``.

    Args:
        products: Pipeline output.
        out_dir: Destination directory (created if needed).

    Returns:
        Paths written.
    """
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    return [
        write_geotiff(
            folder / f"{layer.name}.tif",
            layer.data,
            products.grid,
            layer.dtype,
            layer.nodata,
            layer.description,
        )
        for layer in raster_layers(products)
    ]


def _lonlat(grid: Grid, xy: FloatArray) -> list[list[float]]:
    lon, lat = transform_points(
        CRS.from_user_input(grid.crs), CRS.from_epsg(4326), xy[:, 0].tolist(), xy[:, 1].tolist()
    )
    return [
        [round(a, COORDINATE_DECIMALS), round(b, COORDINATE_DECIMALS)]
        for a, b in zip(lon, lat, strict=True)
    ]


def drainage_geojson(segments: Sequence[DrainageSegment], grid: Grid) -> dict[str, Any]:
    """Drainage reaches as a longitude/latitude GeoJSON FeatureCollection.

    Args:
        segments: Reaches from the drainage network.
        grid: Grid giving the CRS.

    Returns:
        A FeatureCollection of LineStrings with order, length and contributing area.
    """
    features = [
        {
            "type": "Feature",
            "properties": {
                "reach_id": i + 1,
                "strahler_order": seg.strahler_order,
                "length_m": round(seg.length_m, 1),
                "contributing_area_ha": round(seg.contributing_area_ha, 3),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": _lonlat(grid, simplify_staircase(seg.xy)),
            },
        }
        for i, seg in enumerate(segments)
    ]
    return {"type": "FeatureCollection", "features": features}


def zones_geojson(
    zones: Sequence[Zone], summaries: Sequence[ZoneSummary], grid: Grid
) -> dict[str, Any]:
    """Zones with their acreage table as properties, in longitude/latitude.

    Args:
        zones: Zones in the grid CRS.
        summaries: Matching per-zone summaries.
        grid: Grid giving the CRS.

    Returns:
        A FeatureCollection of polygons.
    """
    features = []
    source = CRS.from_user_input(grid.crs)
    for zone, summary in zip(zones, summaries, strict=True):
        properties: dict[str, Any] = {
            "zone_id": zone.zone_id,
            "name": zone.name,
            "total_acres": round(summary.total_acres, 3),
            "woody_acres": round(summary.woody_acres, 3),
            "treatment_acres": round(summary.treatment_acres, 3),
            "mean_slope_deg": round(summary.mean_slope_deg, 2),
        }
        for area in summary.classes:
            key = area.name.lower().replace(" ", "_")
            properties[f"{key}_acres"] = round(area.acres, 3)
        geometry = dict(transform_geom(source, CRS.from_epsg(4326), zone.geometry))
        geometry["coordinates"] = _round_coordinates(geometry["coordinates"])
        features.append({"type": "Feature", "properties": properties, "geometry": geometry})
    return {"type": "FeatureCollection", "features": features}


def _round_coordinates(value: Any) -> Any:
    """Round nested GeoJSON coordinate arrays to ``COORDINATE_DECIMALS``."""
    if isinstance(value, (int, float)):
        return round(float(value), COORDINATE_DECIMALS)
    return [_round_coordinates(item) for item in value]


def write_geojson(path: str | Path, collection: dict[str, Any]) -> Path:
    """Write a GeoJSON FeatureCollection compactly (one feature per line).

    Args:
        path: Output file.
        collection: FeatureCollection.

    Returns:
        The written path.
    """
    out = Path(path)
    lines = [json.dumps(f, separators=(",", ":")) for f in collection["features"]]
    body = ",\n".join(lines)
    out.write_text(
        '{"type":"FeatureCollection","features":[\n' + body + "\n]}\n",
        encoding="utf-8",
        newline="\n",
    )
    return out
