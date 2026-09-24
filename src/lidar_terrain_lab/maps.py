"""Map plates (PNG): terrain, canopy height, vegetation classes, drainage, slope and an overview.

Axes show meters from the south-west corner rather than coordinates, so a plate never
exposes the absolute location of the data it was made from.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from ._types import FloatArray
from .classify import VegClass, woody_floor_m
from .pipeline import SLOPE_CLASS_EDGES_DEG, Products
from .styles import (
    CANOPY_RAMP,
    DRAINAGE_COLOR,
    ELEVATION_RAMP,
    FLOW_RAMP,
    MIN_MAPPED_CONTRIBUTING_AREA_M2,
    PNG_METADATA,
    SLOPE_COLORS,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    ZONE_COLOR,
    ramp,
)

DPI = 110
TRANSPARENT = "#00000000"
PLATE_SIZE_IN = (7.4, 6.6)


def _nice_length(span_m: float) -> float:
    """A 1-2-5 length close to a fifth of the map width, for the scale bar."""
    target = span_m / 5.0
    magnitude = 10.0 ** math.floor(math.log10(target))
    return min((m * magnitude for m in (1, 2, 5, 10)), key=lambda v: abs(v - target))


def _plate(title: str, subtitle: str, footer: str) -> tuple[Figure, Axes]:
    fig = Figure(figsize=PLATE_SIZE_IN, dpi=DPI, facecolor="white")
    ax = fig.add_axes((0.04, 0.10, 0.66, 0.76))
    fig.text(0.04, 0.955, title, fontsize=14, fontweight="bold", color=TEXT_PRIMARY)
    fig.text(0.04, 0.915, subtitle, fontsize=9.5, color=TEXT_SECONDARY)
    fig.text(0.04, 0.035, footer, fontsize=7.5, color=TEXT_SECONDARY, wrap=True)
    return fig, ax


def _decorate(ax: Axes, products: Products) -> None:
    """Scale bar, north arrow and a clean frame."""
    width_m = products.grid.width * products.grid.res_m
    height_m = products.grid.height * products.grid.res_m
    ax.set_xlim(0, width_m)
    ax.set_ylim(0, height_m)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(TEXT_SECONDARY)
        spine.set_linewidth(0.6)
    bar = _nice_length(width_m)
    x0, y0 = 0.04 * width_m, 0.04 * height_m
    ax.add_patch(
        Rectangle((x0, y0), bar, 0.012 * height_m, facecolor=TEXT_PRIMARY, edgecolor="white")
    )
    ax.text(
        x0 + bar / 2,
        y0 + 0.045 * height_m,
        f"{bar:g} m",
        ha="center",
        fontsize=8,
        color=TEXT_PRIMARY,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1},
    )
    ax.annotate(
        "N",
        xy=(0.95, 0.95),
        xytext=(0.95, 0.86),
        xycoords="axes fraction",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=TEXT_PRIMARY,
        arrowprops={"arrowstyle": "-|>", "color": TEXT_PRIMARY, "lw": 1.4},
    )


def _hillshade_base(ax: Axes, products: Products, alpha: float = 1.0) -> None:
    ax.imshow(
        0.35 + 0.65 * products.hillshade,
        extent=products.grid.extent_m,
        cmap="gray",
        vmin=0,
        vmax=1,
        alpha=alpha,
        interpolation="bilinear",
    )


def _drainage(ax: Axes, products: Products, base_width: float = 0.8) -> None:
    for seg in products.hydrology.segments:
        u, v = products.grid.to_local_meters(seg.xy[:, 0], seg.xy[:, 1])
        ax.plot(
            u,
            v,
            color=DRAINAGE_COLOR,
            lw=base_width + 0.7 * seg.strahler_order,
            solid_capstyle="round",
        )


def _zones(ax: Axes, products: Products) -> None:
    if products.zone_raster is None:
        return
    extent = products.grid.extent_m
    ax.contour(
        products.zone_raster,
        levels=np.arange(0.5, len(products.zones) + 1),
        extent=extent,
        origin="upper",
        colors=ZONE_COLOR,
        linewidths=1.2,
    )
    for zone in products.zones:
        zone_rows, zone_cols = np.nonzero(products.zone_raster == zone.zone_id)
        if zone_rows.size == 0:
            continue
        u = (float(zone_cols.mean()) + 0.5) * products.grid.res_m
        v = extent[3] - (float(zone_rows.mean()) + 0.5) * products.grid.res_m
        ax.text(
            u,
            v,
            zone.name,
            ha="center",
            va="center",
            fontsize=8.5,
            color=TEXT_PRIMARY,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": ZONE_COLOR, "pad": 2},
        )


def _shaded_elevation(products: Products) -> FloatArray:
    dem = products.dem
    scaled = (dem - dem.min()) / max(float(np.ptp(dem)), 1e-9)
    rgb = np.asarray(ramp("elevation", ELEVATION_RAMP)(scaled), dtype=np.float64)[..., :3]
    return np.clip(rgb * (0.45 + 0.55 * products.hillshade[..., None]), 0.0, 1.0)


def _log_contributing_area(products: Products) -> FloatArray:
    """log10 of contributing area in square meters, NaN below the mapping floor."""
    area = products.hydrology.flow_accumulation * products.grid.cell_area_m2
    return np.where(area >= MIN_MAPPED_CONTRIBUTING_AREA_M2, np.log10(area), np.nan)


def _class_colors(classes: list[VegClass]) -> tuple[ListedColormap, BoundaryNorm]:
    """Colormap for class codes: woody classes in color, the rest transparent."""
    cmap = ListedColormap([c.color if c.woody else TRANSPARENT for c in classes])
    return cmap, BoundaryNorm(np.arange(0.5, len(classes) + 1.5), cmap.N)


def _save(fig: Figure, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, facecolor="white", metadata=PNG_METADATA)
    return path


def terrain_map(products: Products, path: Path, title_prefix: str, footer: str) -> Path:
    """Shaded bare-earth elevation with drainage lines."""
    stats = products.stats
    fig, ax = _plate(
        f"{title_prefix}: bare-earth terrain",
        f"Elevation {stats.elevation_min_m:.1f} to {stats.elevation_max_m:.1f} m "
        f"({stats.relief_m:.1f} m relief), drainage lines in blue",
        footer,
    )
    ax.imshow(_shaded_elevation(products), extent=products.grid.extent_m, interpolation="bilinear")
    _drainage(ax, products)
    _decorate(ax, products)
    cax = fig.add_axes((0.74, 0.30, 0.025, 0.40))
    image = ax.imshow(
        products.dem,
        extent=products.grid.extent_m,
        cmap=ramp("elevation", ELEVATION_RAMP),
        alpha=0.0,
    )
    bar = fig.colorbar(image, cax=cax)
    if bar.solids is not None:
        bar.solids.set_alpha(1.0)
    bar.set_label("Elevation (m)", color=TEXT_SECONDARY)
    return _save(fig, path)


def canopy_map(products: Products, path: Path, title_prefix: str, footer: str) -> Path:
    """Continuous canopy height over shaded relief."""
    stats = products.stats
    fig, ax = _plate(
        f"{title_prefix}: canopy height",
        f"Surface minus bare earth; tallest crown {stats.canopy_max_m:.1f} m",
        footer,
    )
    _hillshade_base(ax, products)
    floor = woody_floor_m(products.classes)
    chm = np.where(products.chm >= floor, products.chm, np.nan)
    image = ax.imshow(
        chm,
        extent=products.grid.extent_m,
        cmap=ramp("canopy", CANOPY_RAMP),
        vmin=0,
        vmax=max(float(np.nanmax(products.chm)), 1.0),
        interpolation="nearest",
    )
    _decorate(ax, products)
    bar = fig.colorbar(image, cax=fig.add_axes((0.74, 0.30, 0.025, 0.40)))
    bar.set_label(f"Canopy height (m), shown where >= {floor:g} m", color=TEXT_SECONDARY)
    return _save(fig, path)


def vegetation_map(products: Products, path: Path, title_prefix: str, footer: str) -> Path:
    """Height classes with acreage, zone outlines and labels."""
    stats = products.stats
    fig, ax = _plate(
        f"{title_prefix}: vegetation height classes",
        f"Woody cover {stats.woody_acres:.1f} ac ({stats.woody_percent:.0f}%); "
        f"treatment classes {' + '.join(stats.treatment_classes)}: "
        f"{stats.treatment_acres:.1f} ac",
        footer,
    )
    _hillshade_base(ax, products)
    classes = products.classes
    cmap, norm = _class_colors(classes)
    ax.imshow(
        products.veg_codes,
        extent=products.grid.extent_m,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    _zones(ax, products)
    _decorate(ax, products)
    # Non-woody classes are drawn transparent over the relief, so their swatches are unfilled.
    handles: list[Artist] = [
        Patch(
            facecolor=c.color if c.woody else "none",
            edgecolor=TEXT_SECONDARY,
            lw=0.5,
            label=f"{c.label}\n{a.acres:.2f} ac ({a.percent:.1f}%)",
        )
        for c, a in zip(classes, products.class_table, strict=True)
    ]
    if products.zones:
        handles.append(Line2D([0], [0], color=ZONE_COLOR, lw=1.5, label="Zone boundary"))
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.03, 1.0),
        fontsize=8,
        frameon=False,
        labelspacing=1.1,
        title="Class (area)",
        title_fontsize=9,
    )
    return _save(fig, path)


def hydrology_map(products: Products, path: Path, title_prefix: str, footer: str) -> Path:
    """Contributing area and drainage lines by Strahler order."""
    stats = products.stats
    fig, ax = _plate(
        f"{title_prefix}: drainage",
        f"{stats.channel_length_m:.0f} m of drainage line where more than "
        f"{stats.channel_threshold_ha:g} ha drains; largest catchment "
        f"{stats.largest_contributing_area_ha:.1f} ha",
        footer,
    )
    _hillshade_base(ax, products)
    log_area = _log_contributing_area(products)
    image = ax.imshow(
        log_area,
        extent=products.grid.extent_m,
        cmap=ramp("flow", FLOW_RAMP),
        vmin=float(np.nanmin(log_area)),
        vmax=float(np.nanmax(log_area)),
        interpolation="nearest",
    )
    _drainage(ax, products, base_width=0.6)
    _decorate(ax, products)
    bar = fig.colorbar(image, cax=fig.add_axes((0.74, 0.30, 0.025, 0.40)))
    bar.set_label("Contributing area, log10(m2)", color=TEXT_SECONDARY)
    return _save(fig, path)


def slope_map(products: Products, path: Path, title_prefix: str, footer: str) -> Path:
    """Slope in four classes with the share of area in each."""
    stats = products.stats
    fig, ax = _plate(
        f"{title_prefix}: slope",
        f"Mean slope {stats.slope_mean_deg:.1f} degrees",
        footer,
    )
    edges = SLOPE_CLASS_EDGES_DEG
    cmap = ListedColormap(list(SLOPE_COLORS))
    norm = BoundaryNorm(list(edges), cmap.N)
    ax.imshow(
        products.slope, extent=products.grid.extent_m, cmap=cmap, norm=norm, interpolation="nearest"
    )
    _hillshade_base(ax, products, alpha=0.35)
    _decorate(ax, products)
    shares = stats.slope_share_percent
    handles = [
        Patch(
            facecolor=color,
            edgecolor=TEXT_SECONDARY,
            lw=0.5,
            label=f"{label} deg: {shares[label]:.1f}%",
        )
        for color, label in zip(SLOPE_COLORS, shares, strict=True)
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.03, 1.0),
        fontsize=8.5,
        frameon=False,
        title="Slope (share of area)",
        title_fontsize=9,
    )
    return _save(fig, path)


def overview_map(products: Products, path: Path, title_prefix: str, footer: str) -> Path:
    """Four small panels: terrain, canopy classes, drainage and slope."""
    fig = Figure(figsize=(8.0, 8.4), dpi=96, facecolor="white")
    fig.text(
        0.03, 0.965, f"{title_prefix}: overview", fontsize=14, fontweight="bold", color=TEXT_PRIMARY
    )
    fig.text(0.03, 0.02, footer, fontsize=7.5, color=TEXT_SECONDARY)
    extent = products.grid.extent_m
    panels = fig.subplots(
        2,
        2,
        gridspec_kw={
            "left": 0.03,
            "right": 0.97,
            "top": 0.92,
            "bottom": 0.07,
            "wspace": 0.05,
            "hspace": 0.12,
        },
    )
    terrain, classes_ax, drainage, slope = panels.ravel()

    terrain.imshow(_shaded_elevation(products), extent=extent, interpolation="bilinear")
    _drainage(terrain, products, base_width=0.4)
    terrain.set_title("Bare-earth terrain and drainage", fontsize=10, color=TEXT_PRIMARY)

    _hillshade_base(classes_ax, products)
    cmap, norm = _class_colors(products.classes)
    classes_ax.imshow(
        products.veg_codes, extent=extent, cmap=cmap, norm=norm, interpolation="nearest"
    )
    classes_ax.set_title("Shrub / small tree / tree", fontsize=10, color=TEXT_PRIMARY)

    _hillshade_base(drainage, products)
    drainage.imshow(
        _log_contributing_area(products),
        extent=extent,
        cmap=ramp("flow", FLOW_RAMP),
        interpolation="nearest",
    )
    drainage.set_title("Contributing area", fontsize=10, color=TEXT_PRIMARY)

    slope.imshow(
        products.slope,
        extent=extent,
        cmap=ListedColormap(list(SLOPE_COLORS)),
        norm=BoundaryNorm(list(SLOPE_CLASS_EDGES_DEG), len(SLOPE_COLORS)),
        interpolation="nearest",
    )
    slope.set_title("Slope class", fontsize=10, color=TEXT_PRIMARY)

    for ax in panels.ravel():
        _decorate(ax, products)
    return _save(fig, path)


def write_maps(
    products: Products, out_dir: Path, title_prefix: str, footer: str
) -> dict[str, Path]:
    """Render every plate into ``out_dir``.

    Args:
        products: Pipeline output.
        out_dir: Destination directory (created if needed).
        title_prefix: Leading part of each plate title.
        footer: Data-source note printed on every plate.

    Returns:
        Map name to written path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plates = {
        "overview": overview_map,
        "terrain": terrain_map,
        "canopy_height": canopy_map,
        "vegetation_classes": vegetation_map,
        "drainage": hydrology_map,
        "slope": slope_map,
    }
    return {
        name: plot(products, out_dir / f"{name}.png", title_prefix, footer)
        for name, plot in plates.items()
    }
