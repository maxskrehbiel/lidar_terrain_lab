"""Runs every stage in order and collects the rasters, vectors and statistics they produce."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from ._types import BoolArray, FloatArray, Int32Array, IntArray, UInt8Array
from .canopy import canopy_height_model, surface_model
from .classify import (
    ClassArea,
    VegClass,
    build_classes,
    class_areas,
    classify_heights,
    treatment_acres,
    woody_acres,
)
from .config import HydrologyParams, PipelineParams
from .errors import InputDataError
from .ground import bare_earth_dem, classify_ground
from .hydrology import (
    DrainageSegment,
    d8_flow_direction,
    drainage_network,
    flow_accumulation,
    priority_flood_fill,
)
from .point_cloud import PointCloud
from .rasterize import SQUARE_METERS_PER_ACRE, SQUARE_METERS_PER_HECTARE, Grid
from .terrain import aspect_degrees, hillshade, slope_degrees
from .zones import Zone, ZoneSummary, rasterize_zones, summarize_zones, zones_from_geojson

logger = logging.getLogger(__name__)

SLOPE_CLASS_EDGES_DEG = (0.0, 5.0, 10.0, 20.0, 90.0)


@dataclass(frozen=True)
class SiteStats:
    """Scalar statistics for one run, as shown in reports.

    Attributes:
        points_total: Points read, including flagged noise.
        points_noise: Points the provider flagged as noise (dropped).
        points_ground: Points labeled ground.
        ground_method: ``existing`` or ``pmf``.
        point_density_per_m2: Non-noise points per square meter.
        ground_density_per_m2: Ground points per square meter.
        dem_interpolated_percent: Share of DEM cells without a ground return.
        area_acres: Grid area in acres.
        area_hectares: Grid area in hectares.
        resolution_m: Cell size in meters.
        elevation_min_m: Lowest DEM cell.
        elevation_max_m: Highest DEM cell.
        relief_m: Highest minus lowest.
        slope_mean_deg: Mean slope.
        slope_share_percent: Share of area per slope class, keyed by class label.
        canopy_max_m: Tallest canopy height.
        woody_acres: Area of the woody classes.
        woody_percent: Woody share of the grid.
        treatment_classes: Names of the treatment classes.
        treatment_acres: Area of the treatment classes.
        channel_threshold_ha: Contributing area that starts a drainage line.
        channel_length_m: Total drainage-line length.
        channel_reaches: Number of reaches.
        max_strahler_order: Highest Strahler order.
        largest_contributing_area_ha: Largest area draining through one cell.
        sink_acres: Area of filled depressions deeper than the reporting depth.
        max_sink_depth_m: Deepest filled depression.
    """

    points_total: int
    points_noise: int
    points_ground: int
    ground_method: str
    point_density_per_m2: float
    ground_density_per_m2: float
    dem_interpolated_percent: float
    area_acres: float
    area_hectares: float
    resolution_m: float
    elevation_min_m: float
    elevation_max_m: float
    relief_m: float
    slope_mean_deg: float
    slope_share_percent: dict[str, float]
    canopy_max_m: float
    woody_acres: float
    woody_percent: float
    treatment_classes: tuple[str, ...]
    treatment_acres: float
    channel_threshold_ha: float
    channel_length_m: float
    channel_reaches: int
    max_strahler_order: int
    largest_contributing_area_ha: float
    sink_acres: float
    max_sink_depth_m: float


@dataclass(frozen=True)
class Hydrology:
    """Outputs of the surface-water stage.

    Attributes:
        filled_dem: DEM with depressions filled to their spill level (m).
        sink_depth: How much filling raised each cell (m).
        flow_direction: D8 codes.
        flow_accumulation: Upstream cell count including the cell itself.
        channel_mask: Cells whose contributing area passes the channel threshold.
        stream_order: Strahler order on channel cells, 0 elsewhere.
        segments: Drainage reaches as polylines.
    """

    filled_dem: FloatArray
    sink_depth: FloatArray
    flow_direction: UInt8Array
    flow_accumulation: FloatArray
    channel_mask: BoolArray
    stream_order: IntArray
    segments: list[DrainageSegment]


@dataclass(frozen=True)
class Products:
    """Everything one pipeline run produces.

    Attributes:
        grid: The shared raster grid.
        params: Settings used.
        dem: Bare-earth elevation (m).
        dsm: Highest vegetation-eligible surface (m).
        chm: Canopy height, surface minus ground (m).
        slope: Slope (degrees).
        aspect: Aspect (degrees clockwise from north, NaN on flats).
        hillshade: Shaded relief on ``[0, 1]``.
        hydrology: Filled DEM, flow routing and drainage network.
        veg_codes: Vegetation class per cell.
        classes: Vegetation class definitions.
        class_table: Area per class over the whole grid.
        zones: Management zones (empty when none were supplied).
        zone_raster: Zone id per cell, or ``None``.
        zone_table: Per-zone summaries.
        is_ground: Ground flag per input point (noise is never ground).
        stats: Scalar statistics for reports.
    """

    grid: Grid
    params: PipelineParams
    dem: FloatArray
    dsm: FloatArray
    chm: FloatArray
    slope: FloatArray
    aspect: FloatArray
    hillshade: FloatArray
    hydrology: Hydrology
    veg_codes: UInt8Array
    classes: list[VegClass]
    class_table: list[ClassArea]
    zones: list[Zone]
    zone_raster: Int32Array | None
    zone_table: list[ZoneSummary]
    is_ground: BoolArray
    stats: SiteStats


def grid_for(cloud: PointCloud, resolution_m: float) -> Grid:
    """Grid covering the cloud's requested extent (or its points) at a resolution in meters.

    Args:
        cloud: Point cloud with a CRS.
        resolution_m: Cell size in meters.

    Returns:
        A snapped grid in the cloud's CRS.
    """
    west, south, east, north = cloud.extent or cloud.bounds
    return Grid.from_bounds(
        west,
        south,
        east,
        north,
        resolution_m / cloud.meters_per_unit,
        cloud.crs,
        cloud.meters_per_unit,
    )


def route_water(dem: FloatArray, grid: Grid, params: HydrologyParams) -> Hydrology:
    """Fill depressions, route flow with D8, accumulate it and trace drainage lines.

    Args:
        dem: Bare-earth DEM.
        grid: Grid for cell size and coordinates.
        params: Drainage settings.

    Returns:
        The hydrology products.
    """
    filled = priority_flood_fill(dem)
    direction = d8_flow_direction(filled, grid.res_m)
    accumulation = flow_accumulation(direction)
    threshold_cells = params.channel_threshold_ha * SQUARE_METERS_PER_HECTARE / grid.cell_area_m2
    channels, order, segments = drainage_network(direction, accumulation, grid, threshold_cells)
    logger.info("hydrology: %d drainage reaches", len(segments))
    return Hydrology(
        filled_dem=filled,
        sink_depth=np.clip(filled - dem, 0.0, None),
        flow_direction=direction,
        flow_accumulation=accumulation,
        channel_mask=channels,
        stream_order=order,
        segments=segments,
    )


def _slope_shares(slope: FloatArray) -> dict[str, float]:
    shares = {}
    for low, high in pairwise(SLOPE_CLASS_EDGES_DEG):
        label = f"{low:g}-{high:g}" if high < SLOPE_CLASS_EDGES_DEG[-1] else f">{low:g}"
        shares[label] = 100.0 * int(((slope >= low) & (slope < high)).sum()) / slope.size
    return shares


@dataclass(frozen=True)
class _Surfaces:
    """The rasters the statistics are computed from."""

    dem: FloatArray
    chm: FloatArray
    slope: FloatArray
    hydrology: Hydrology


def _site_stats(
    grid: Grid,
    params: PipelineParams,
    counts: tuple[int, int, int],
    ground: tuple[str, float],
    surfaces: _Surfaces,
    table: list[ClassArea],
) -> SiteStats:
    """Summary numbers; ``counts`` is (total, noise, ground) and ``ground`` is (method, voids)."""
    total, noise, ground_points = counts
    method, void_fraction = ground
    dem, hydro = surfaces.dem, surfaces.hydrology
    area_m2 = grid.width * grid.height * grid.cell_area_m2
    sinks = int((hydro.sink_depth > params.hydrology.sink_report_depth_m).sum())
    return SiteStats(
        points_total=total,
        points_noise=noise,
        points_ground=ground_points,
        ground_method=method,
        point_density_per_m2=(total - noise) / area_m2,
        ground_density_per_m2=ground_points / area_m2,
        dem_interpolated_percent=100.0 * void_fraction,
        area_acres=area_m2 / SQUARE_METERS_PER_ACRE,
        area_hectares=area_m2 / SQUARE_METERS_PER_HECTARE,
        resolution_m=grid.res_m,
        elevation_min_m=float(dem.min()),
        elevation_max_m=float(dem.max()),
        relief_m=float(dem.max() - dem.min()),
        slope_mean_deg=float(surfaces.slope.mean()),
        slope_share_percent=_slope_shares(surfaces.slope),
        canopy_max_m=float(surfaces.chm.max()),
        woody_acres=woody_acres(table),
        woody_percent=sum(a.percent for a in table if a.woody),
        treatment_classes=tuple(a.name for a in table if a.treatment),
        treatment_acres=treatment_acres(table),
        channel_threshold_ha=params.hydrology.channel_threshold_ha,
        channel_length_m=sum(s.length_m for s in hydro.segments),
        channel_reaches=len(hydro.segments),
        max_strahler_order=int(hydro.stream_order.max()),
        largest_contributing_area_ha=float(hydro.flow_accumulation.max())
        * grid.cell_area_m2
        / SQUARE_METERS_PER_HECTARE,
        sink_acres=sinks * grid.cell_area_m2 / SQUARE_METERS_PER_ACRE,
        max_sink_depth_m=float(hydro.sink_depth.max()),
    )


@dataclass(frozen=True)
class _Ground:
    """Ground labels and the DEM built from them."""

    is_ground: BoolArray
    dem: FloatArray
    void_fraction: float
    method: str
    count: int


def _ground_stage(
    cloud: PointCloud, noise: BoolArray, grid: Grid, params: PipelineParams
) -> _Ground:
    points = cloud.subset(~noise)
    ground = classify_ground(points, params.ground, params.resolution_m)
    is_ground = np.zeros(len(cloud), dtype=bool)
    is_ground[~noise] = ground.is_ground
    g = ground.is_ground
    dem, void_fraction = bare_earth_dem(grid, points.x[g], points.y[g], points.z[g])
    logger.info("DEM built; %.1f%% of cells interpolated", 100 * void_fraction)
    return _Ground(is_ground, dem, void_fraction, ground.method, int(g.sum()))


def _zone_stage(
    zone_collection: dict[str, Any] | None,
    grid: Grid,
    codes: UInt8Array,
    classes: list[VegClass],
    slope: FloatArray,
) -> tuple[list[Zone], Int32Array | None, list[ZoneSummary]]:
    if zone_collection is None:
        return [], None, []
    zones = zones_from_geojson(zone_collection, grid)
    raster = rasterize_zones(zones, grid)
    return zones, raster, summarize_zones(raster, zones, codes, classes, slope, grid.cell_area_m2)


def run_pipeline(
    cloud: PointCloud,
    params: PipelineParams,
    zone_collection: dict[str, Any] | None = None,
    grid: Grid | None = None,
) -> Products:
    """Turn a point cloud into terrain, canopy, drainage and vegetation products.

    Args:
        cloud: Input points (ground classification optional).
        params: Pipeline settings.
        zone_collection: Optional GeoJSON FeatureCollection of management zones.
        grid: Optional output grid; defaults to the cloud's extent at ``params.resolution_m``.

    Returns:
        All products and summary statistics.

    Raises:
        InputDataError: If the cloud has no CRS or no usable points.
    """
    if cloud.crs is None:
        raise InputDataError("the point cloud needs a CRS")
    noise = cloud.noise_mask()
    if noise.all():
        raise InputDataError("no points left after removing noise")
    grid = grid or grid_for(cloud, params.resolution_m)
    logger.info("%d points (%d flagged noise), grid %dx%d", len(cloud), noise.sum(), *grid.shape)

    ground = _ground_stage(cloud, noise, grid, params)
    dem = ground.dem
    kept = ~noise
    dsm = surface_model(
        grid, cloud.x[kept], cloud.y[kept], cloud.z[kept], cloud.classification[kept], dem
    )
    chm = canopy_height_model(dsm, dem, params.canopy)
    slope = slope_degrees(dem, grid.res_m)
    hydro = route_water(dem, grid, params.hydrology)
    classes = build_classes(params.vegetation)
    codes = classify_heights(chm, classes)
    table = class_areas(codes, classes, grid.cell_area_m2)
    zones, zone_raster, zone_table = _zone_stage(zone_collection, grid, codes, classes, slope)
    stats = _site_stats(
        grid,
        params,
        (len(cloud), int(noise.sum()), ground.count),
        (ground.method, ground.void_fraction),
        _Surfaces(dem, chm, slope, hydro),
        table,
    )
    return Products(
        grid=grid,
        params=params,
        dem=dem,
        dsm=dsm,
        chm=chm,
        slope=slope,
        aspect=aspect_degrees(dem, grid.res_m),
        hillshade=hillshade(dem, grid.res_m, params.hillshade),
        hydrology=hydro,
        veg_codes=codes,
        classes=classes,
        class_table=table,
        zones=zones,
        zone_raster=zone_raster,
        zone_table=zone_table,
        is_ground=ground.is_ground,
        stats=stats,
    )
