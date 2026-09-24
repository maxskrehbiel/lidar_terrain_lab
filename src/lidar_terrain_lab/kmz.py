"""Google Earth KMZ packaging: raster ground overlays, drainage lines, zones and a legend.

KML is plain XML, so the archive is assembled with the standard library (``zipfile``).
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from xml.sax.saxutils import escape

import matplotlib.image
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

from ._types import FloatArray
from .classify import ClassArea, VegClass, woody_floor_m
from .pipeline import Products
from .rasterize import Grid
from .styles import (
    CANOPY_RAMP,
    DRAINAGE_COLOR,
    FLOW_RAMP,
    MIN_MAPPED_CONTRIBUTING_AREA_M2,
    PNG_METADATA,
    SLOPE_COLORS,
    TEXT_SECONDARY,
    ZONE_COLOR,
    ramp,
)

# Zip entries carry a modification time; fixing it keeps re-runs byte-identical.
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
OVERLAY_OPACITY = 0.85
SLOPE_OVERLAY_MAX_DEG = 30.0
STYLED_ORDERS = 4  # drainage styles exist for orders 1-4; higher orders reuse the last


@dataclass(frozen=True)
class _Overlay:
    name: str
    data: FloatArray
    colorize: Callable[[FloatArray], FloatArray]
    resampling: Resampling
    visible: bool


def _kml_color(hex_color: str, alpha: float = 1.0) -> str:
    """KML wants ``aabbggrr``."""
    r, g, b, _ = to_rgba(hex_color)
    return f"{round(alpha * 255):02x}{round(b * 255):02x}{round(g * 255):02x}{round(r * 255):02x}"


def _to_wgs84(
    data: FloatArray, grid: Grid, resampling: Resampling
) -> tuple[FloatArray, tuple[float, float, float, float]]:
    """Reproject a raster to longitude/latitude for a Google Earth ground overlay."""
    source_crs = CRS.from_user_input(grid.crs)
    dst_transform, width, height = calculate_default_transform(
        source_crs, CRS.from_epsg(4326), grid.width, grid.height, *grid.bounds
    )
    out = np.full((height, width), np.nan)
    reproject(
        data.astype(np.float64),
        out,
        src_transform=grid.transform,
        src_crs=source_crs,
        dst_transform=dst_transform,
        dst_crs=CRS.from_epsg(4326),
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    west, north = dst_transform.c, dst_transform.f
    east = west + dst_transform.a * width
    south = north + dst_transform.e * height
    return out, (float(west), float(south), float(east), float(north))


def _png_bytes(rgba: FloatArray) -> bytes:
    buffer = io.BytesIO()
    # imsave skips None values at runtime (that is what drops the tag); its stub is narrower.
    metadata = cast(dict[str, str], PNG_METADATA)
    matplotlib.image.imsave(buffer, np.clip(rgba, 0.0, 1.0), format="png", metadata=metadata)
    return buffer.getvalue()


def _class_rgba(codes: FloatArray, classes: Sequence[VegClass]) -> FloatArray:
    rgba = np.zeros((*codes.shape, 4))
    for veg in classes:
        if veg.woody:  # non-woody classes stay transparent so imagery shows through
            rgba[codes == veg.code] = to_rgba(veg.color)
    return rgba


def _ramp_rgba(values: FloatArray, colors: tuple[str, ...], vmin: float, vmax: float) -> FloatArray:
    scaled = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = np.asarray(ramp("overlay", colors)(np.nan_to_num(scaled)), dtype=np.float64)
    rgba[..., 3] = np.where(np.isfinite(values), OVERLAY_OPACITY, 0.0)
    return rgba


def _legend_png(classes: Sequence[VegClass], areas: Sequence[ClassArea]) -> bytes:
    fig = Figure(figsize=(2.6, 1.3), dpi=110)
    handles = [
        Patch(facecolor=c.color, edgecolor=TEXT_SECONDARY, label=f"{c.label}: {a.acres:.2f} ac")
        for c, a in zip(classes, areas, strict=True)
    ]
    fig.legend(handles=handles, loc="center", frameon=True, fontsize=7, title="Vegetation height")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor="white", metadata=PNG_METADATA)
    return buffer.getvalue()


def _overlays(products: Products) -> list[_Overlay]:
    """The raster layers, in drawing order."""
    classes = products.classes
    chm_top = max(float(np.nanmax(products.chm)), 1.0)
    area = products.hydrology.flow_accumulation * products.grid.cell_area_m2
    log_area = np.where(area >= MIN_MAPPED_CONTRIBUTING_AREA_M2, np.log10(area), np.nan)
    flow_range = (float(np.nanmin(log_area)), float(np.nanmax(log_area)))
    woody_floor = woody_floor_m(classes)
    return [
        _Overlay(
            "Shaded relief",
            products.hillshade,
            lambda a: np.dstack([a, a, a, np.isfinite(a).astype(float)]),
            Resampling.bilinear,
            True,
        ),
        _Overlay(
            "Vegetation height class",
            products.veg_codes.astype(np.float64),
            lambda a: _class_rgba(a, classes),
            Resampling.nearest,
            True,
        ),
        _Overlay(
            "Canopy height (m)",
            np.where(products.chm >= woody_floor, products.chm, np.nan),
            lambda a: _ramp_rgba(a, CANOPY_RAMP, 0.0, chm_top),
            Resampling.bilinear,
            False,
        ),
        _Overlay(
            "Contributing area (log10 m2)",
            log_area,
            lambda a: _ramp_rgba(a, FLOW_RAMP, *flow_range),
            Resampling.bilinear,
            False,
        ),
        _Overlay(
            "Slope (degrees)",
            products.slope,
            lambda a: _ramp_rgba(a, SLOPE_COLORS, 0.0, SLOPE_OVERLAY_MAX_DEG),
            Resampling.bilinear,
            False,
        ),
    ]


def _overlay_kml(products: Products) -> tuple[str, dict[str, bytes]]:
    """GroundOverlay elements plus the PNG files they reference."""
    files: dict[str, bytes] = {}
    elements = []
    for i, overlay in enumerate(_overlays(products)):
        data, (west, south, east, north) = _to_wgs84(
            overlay.data, products.grid, overlay.resampling
        )
        name = f"overlays/layer_{i}.png"
        files[name] = _png_bytes(overlay.colorize(data))
        elements.append(
            f"<GroundOverlay><name>{escape(overlay.name)}</name>"
            f"<visibility>{int(overlay.visible)}</visibility><drawOrder>{i}</drawOrder>"
            f"<Icon><href>{name}</href></Icon><LatLonBox><north>{north}</north>"
            f"<south>{south}</south><east>{east}</east><west>{west}</west></LatLonBox>"
            "</GroundOverlay>"
        )
    files["overlays/legend.png"] = _legend_png(products.classes, products.class_table)
    return "".join(elements), files


def _coordinates(ring: Sequence[Sequence[float]]) -> str:
    return " ".join(f"{lon},{lat},0" for lon, lat in ring)


def _drainage_kml(drainage: dict[str, Any]) -> str:
    placemarks = []
    for feature in drainage["features"]:
        props = feature["properties"]
        order = int(props["strahler_order"])
        placemarks.append(
            f"<Placemark><name>Reach {props['reach_id']} (order {order})</name>"
            f"<description>{props['length_m']} m long; drains "
            f"{props['contributing_area_ha']} ha</description>"
            f"<styleUrl>#drainage_{min(order, STYLED_ORDERS)}</styleUrl>"
            "<LineString><tessellate>1</tessellate>"
            f"<coordinates>{_coordinates(feature['geometry']['coordinates'])}</coordinates>"
            "</LineString></Placemark>"
        )
    return "".join(placemarks)


def _zone_kml(zones: dict[str, Any] | None) -> str:
    placemarks = []
    for feature in (zones or {}).get("features", []):
        props = feature["properties"]
        rows = "".join(
            f"<tr><td>{escape(key.replace('_', ' '))}</td><td>{value}</td></tr>"
            for key, value in props.items()
            if key not in ("zone_id", "name")
        )
        geometry = feature["geometry"]
        polygons = (
            [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        )
        shapes = "".join(
            "<Polygon><tessellate>1</tessellate><outerBoundaryIs><LinearRing><coordinates>"
            f"{_coordinates(polygon[0])}</coordinates></LinearRing></outerBoundaryIs></Polygon>"
            for polygon in polygons
        )
        placemarks.append(
            f"<Placemark><name>{escape(props['name'])}</name>"
            f"<description><![CDATA[<table>{rows}</table>]]></description>"
            f"<styleUrl>#zone</styleUrl><MultiGeometry>{shapes}</MultiGeometry></Placemark>"
        )
    return "".join(placemarks)


def _styles_kml() -> str:
    drainage = "".join(
        f'<Style id="drainage_{order}"><LineStyle><color>{_kml_color(DRAINAGE_COLOR)}</color>'
        f"<width>{1 + order}</width></LineStyle></Style>"
        for order in range(1, STYLED_ORDERS + 1)
    )
    zone = (
        f'<Style id="zone"><LineStyle><color>{_kml_color(ZONE_COLOR)}</color>'
        "<width>2</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>"
    )
    return drainage + zone


def _legend_kml() -> str:
    return (
        "<ScreenOverlay><name>Legend</name><Icon><href>overlays/legend.png</href></Icon>"
        '<overlayXY x="0" y="0" xunits="fraction" yunits="fraction"/>'
        '<screenXY x="0.01" y="0.04" xunits="fraction" yunits="fraction"/>'
        '<size x="0" y="0" xunits="fraction" yunits="fraction"/></ScreenOverlay>'
    )


def _write_zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in entries.items():
            info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)
    return path


def write_kmz(
    path: str | Path,
    products: Products,
    title: str,
    description: str,
    drainage: dict[str, Any],
    zones: dict[str, Any] | None,
) -> Path:
    """Package ground overlays, drainage lines and zones for Google Earth.

    Args:
        path: Output ``.kmz`` file.
        products: Pipeline output.
        title: Document name.
        description: Document description (plain text).
        drainage: Drainage lines as a longitude/latitude FeatureCollection.
        zones: Zone summaries as a longitude/latitude FeatureCollection, or ``None``.

    Returns:
        The written path.
    """
    overlays, files = _overlay_kml(products)
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        f"<name>{escape(title)}</name><description>{escape(description)}</description>"
        f"{_styles_kml()}"
        f"<Folder><name>Rasters</name>{overlays}</Folder>"
        f"<Folder><name>Drainage lines</name>{_drainage_kml(drainage)}</Folder>"
        f"<Folder><name>Zones</name>{_zone_kml(zones)}</Folder>"
        f"{_legend_kml()}</Document></kml>\n"
    )
    return _write_zip(Path(path), {"doc.kml": kml.encode("utf-8"), **files})
