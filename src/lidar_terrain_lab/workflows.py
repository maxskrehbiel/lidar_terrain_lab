"""End-to-end runs behind the CLI: the offline synthetic demo and a run on real LAZ tiles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from .config import GroundMethod, PipelineParams, RegionConfig
from .export import drainage_geojson, write_geojson, write_raster_bundle, zones_geojson
from .kmz import write_kmz
from .maps import write_maps
from .pipeline import Products, run_pipeline
from .point_cloud import ZUnits, read_las
from .rasterize import SQUARE_METERS_PER_HECTARE
from .report import (
    RunMetadata,
    build_summary,
    write_html,
    write_markdown,
    write_summary_json,
)
from .synthetic import SceneParams, SyntheticScene, make_scene
from .validation import Check, validate
from .zones import read_feature_collection

DEMO_CHANNEL_SHARE = 0.03  # drainage lines start where 3% of the scene drains through a cell


@dataclass(frozen=True)
class OutputPaths:
    """Files one run wrote.

    Attributes:
        report_md: Markdown report.
        report_html: HTML report.
        summary_json: Machine-readable summary.
        maps: Map name to PNG.
        drainage_geojson: Drainage lines.
        zones_geojson: Zone summaries, if zones were given.
        kmz: Google Earth package.
        geotiffs: GeoTIFF bundle (empty when skipped).
    """

    report_md: Path
    report_html: Path
    summary_json: Path
    maps: dict[str, Path]
    drainage_geojson: Path
    zones_geojson: Path | None
    kmz: Path
    geotiffs: list[Path]


def write_outputs(
    products: Products,
    out_dir: Path,
    meta: RunMetadata,
    footer: str,
    checks: Sequence[Check] | None = None,
    write_geotiffs: bool = True,
) -> OutputPaths:
    """Write maps, vectors, KMZ, GeoTIFFs and reports for one run.

    Args:
        products: Pipeline output.
        out_dir: Destination directory.
        meta: Report metadata.
        footer: Data-source note printed on every map.
        checks: Validation checks to include (synthetic runs).
        write_geotiffs: Whether to write the GeoTIFF bundle.

    Returns:
        Paths of everything written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors = out_dir / "vectors"
    vectors.mkdir(exist_ok=True)
    maps = write_maps(products, out_dir / "maps", meta.title, footer)
    drainage = drainage_geojson(products.hydrology.segments, products.grid)
    drainage_path = write_geojson(vectors / "drainage_lines.geojson", drainage)
    zones_fc = None
    zones_path = None
    if products.zones:
        zones_fc = zones_geojson(products.zones, products.zone_table, products.grid)
        zones_path = write_geojson(vectors / "zones_summary.geojson", zones_fc)
    kmz = write_kmz(
        out_dir / "google_earth.kmz",
        products,
        meta.title,
        f"{meta.data_label}. {meta.source}",
        drainage,
        zones_fc,
    )
    geotiffs = write_raster_bundle(products, out_dir / "geotiff") if write_geotiffs else []
    summary = build_summary(products, meta, checks)
    return OutputPaths(
        report_md=write_markdown(summary, maps, out_dir / "report.md"),
        report_html=write_html(summary, maps, out_dir / "report.html"),
        summary_json=write_summary_json(summary, out_dir / "summary.json"),
        maps=maps,
        drainage_geojson=drainage_path,
        zones_geojson=zones_path,
        kmz=kmz,
        geotiffs=geotiffs,
    )


@dataclass(frozen=True)
class DemoResult:
    """Outcome of the synthetic demo.

    Attributes:
        scene: The generated scene (with truth).
        products: Pipeline output.
        checks: Validation against truth.
        outputs: Files written.
    """

    scene: SyntheticScene
    products: Products
    checks: list[Check]
    outputs: OutputPaths


def run_demo(
    out_dir: Path,
    seed: int = 7,
    scene_params: SceneParams | None = None,
    write_geotiffs: bool = True,
) -> DemoResult:
    """Generate a synthetic scene, run the pipeline, score it against truth and write outputs.

    Args:
        out_dir: Destination directory.
        seed: Scene seed.
        scene_params: Scene settings.
        write_geotiffs: Whether to write the GeoTIFF bundle.

    Returns:
        The scene, products, validation checks and output paths.
    """
    scene = make_scene(seed, scene_params)
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors = out_dir / "vectors"
    vectors.mkdir(exist_ok=True)
    zones_input = vectors / "zones_input.geojson"
    write_geojson(zones_input, scene.zones)

    base = PipelineParams(resolution_m=scene.params.res_m)
    scene_ha = scene.params.size_m**2 / SQUARE_METERS_PER_HECTARE
    params = replace(
        base,
        hydrology=replace(base.hydrology, channel_threshold_ha=DEMO_CHANNEL_SHARE * scene_ha),
    )
    # Read the zones back from disk so the demo exercises the same path a user's file takes.
    zones = read_feature_collection(zones_input)
    products = run_pipeline(scene.cloud, params, zones, grid=scene.grid)
    checks = validate(products, scene)
    size = scene.params.size_m
    meta = RunMetadata(
        title="Synthetic demo",
        data_label="SYNTHETIC",
        source=(
            f"Generated point cloud (seed {seed}): a {size:g} m x {size:g} m scene with rolling "
            "terrain, a meandering channel, a tributary, and scattered shrubs and trees."
        ),
        notes=(
            "Not a real place: the scene is georeferenced beside 0 N, 0 E, in open ocean, "
            "so it cannot be confused with real land."
        ),
    )
    footer = f"SYNTHETIC DATA (seed {seed}). Grid north up; distances in meters."
    outputs = write_outputs(products, out_dir, meta, footer, checks, write_geotiffs)
    return DemoResult(scene, products, checks, outputs)


def run_on_tiles(
    laz_paths: Sequence[Path],
    out_dir: Path,
    params: PipelineParams,
    region: RegionConfig | None = None,
    zones_path: Path | None = None,
    crs: str | None = None,
    z_units: ZUnits = "auto",
    ground_method: GroundMethod | None = None,
    write_geotiffs: bool = True,
) -> tuple[Products, OutputPaths]:
    """Run the pipeline on LAZ tiles (for example public 3DEP tiles from ``fetch``).

    Args:
        laz_paths: Input tiles.
        out_dir: Destination directory.
        params: Pipeline settings.
        region: Optional region; its box crops the tiles and its name titles the report.
        zones_path: Optional GeoJSON of management zones.
        crs: CRS override for tiles without one.
        z_units: Vertical units override.
        ground_method: Override for ``params.ground.method``.
        write_geotiffs: Whether to write the GeoTIFF bundle.

    Returns:
        Products and output paths.
    """
    if ground_method is not None:
        params = replace(params, ground=replace(params.ground, method=ground_method))
    cloud = read_las(
        laz_paths,
        bbox_wgs84=region.bbox_wgs84 if region else None,
        crs=crs,
        z_units=z_units,
    )
    zones = read_feature_collection(zones_path) if zones_path is not None else None
    products = run_pipeline(cloud, params, zones)
    names = ", ".join(cloud.sources)
    meta = RunMetadata(
        title=region.name if region else "LiDAR terrain report",
        data_label="USGS 3DEP LiDAR" if region else "LiDAR",
        source=f"Point cloud tiles: {names}.",
        notes=region.land_status if region else "",
    )
    source_note = "USGS 3DEP LiDAR (public domain)" if region else "user-supplied LiDAR"
    footer = f"Source: {source_note}. Grid north up; distances in meters."
    outputs = write_outputs(products, out_dir, meta, footer, None, write_geotiffs)
    return products, outputs
