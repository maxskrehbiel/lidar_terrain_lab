"""Command-line interface: ``demo``, ``fetch`` and ``run`` subcommands."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from . import __version__
from .config import (
    Config,
    GroundMethod,
    PipelineParams,
    RegionConfig,
    load_config,
    load_demo_region_config,
)
from .errors import ConfigError, LidarDependencyError, LidarTerrainLabError
from .fetch import LidarTile, download_tiles, search_tiles, select_tiles
from .point_cloud import ZUnits
from .synthetic import SceneParams
from .workflows import run_demo, run_on_tiles

DEFAULT_DEMO_DIR = Path("demo_output")
DEFAULT_DOWNLOAD_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("outputs")
LARGE_DOWNLOAD_MB = 1000.0
EXIT_OK = 0
EXIT_CHECKS_FAILED = 1
EXIT_USAGE = 2
EXIT_MISSING_DEPENDENCY = 3
LOG_LEVELS = (logging.WARNING, logging.INFO, logging.DEBUG)


def non_negative_int(text: str) -> int:
    """Argparse type for a count or seed that may be zero."""
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from exc
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be zero or more, got {value}")
    return value


def positive_float(text: str) -> float:
    """Argparse type for a size that must be greater than zero."""
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from exc
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be greater than zero, got {text}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line interface.

    Returns:
        The argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="lidar_terrain_lab",
        description="LiDAR point cloud to terrain, canopy height, drainage and vegetation acreage.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="-v for progress, -vv for debug detail"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="run offline on a seeded synthetic scene")
    demo.add_argument("--out", type=Path, default=DEFAULT_DEMO_DIR, help="output directory")
    demo.add_argument("--seed", type=non_negative_int, default=7, help="scene seed")
    demo.add_argument("--size-m", type=positive_float, default=400.0, help="scene side in meters")
    demo.add_argument("--no-geotiff", action="store_true", help="skip the GeoTIFF bundle")

    fetch = commands.add_parser("fetch", help="download public USGS 3DEP tiles for a region")
    fetch.add_argument("--config", type=Path, help="TOML with [region] (default: demo region)")
    fetch.add_argument("--dest", type=Path, default=DEFAULT_DOWNLOAD_DIR, help="download folder")
    fetch.add_argument("--list-only", action="store_true", help="list tiles without downloading")
    fetch.add_argument("--yes", action="store_true", help="allow downloads over 1 GB")

    run = commands.add_parser("run", help="process LAZ tiles (local files or --fetch)")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--laz", type=Path, nargs="+", help="local LAS/LAZ files")
    source.add_argument("--fetch", action="store_true", help="download the region's 3DEP tiles")
    run.add_argument(
        "--config",
        type=Path,
        help="TOML settings (default: built-in settings; with --fetch, the demo region)",
    )
    run.add_argument("--dest", type=Path, default=DEFAULT_DOWNLOAD_DIR, help="download folder")
    run.add_argument("--zones", type=Path, help="GeoJSON polygons with a 'name' property")
    run.add_argument("--out", type=Path, help="output directory (default outputs/<region>)")
    run.add_argument("--ground", choices=["auto", "existing", "pmf"], help="ground method")
    run.add_argument("--crs", help="CRS for tiles whose header has none, e.g. EPSG:26914")
    run.add_argument("--z-units", choices=["auto", "m", "ft", "us-ft"], default="auto")
    run.add_argument("--no-region-crop", action="store_true", help="process whole tiles")
    run.add_argument("--no-geotiff", action="store_true", help="skip the GeoTIFF bundle")
    run.add_argument("--yes", action="store_true", help="allow downloads over 1 GB")
    return parser


def _slug(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in cleaned.split("_") if part)[:60] or "run"


def _region_config(path: Path | None) -> tuple[Config, RegionConfig]:
    config = load_demo_region_config() if path is None else load_config(path)
    if config.region is None:
        raise ConfigError(f"{path} has no [region] table")
    return config, config.region


def _print_tiles(tiles: Sequence[LidarTile]) -> float:
    total_mb = sum(t.size_bytes for t in tiles) / 1e6
    print(f"{len(tiles)} tile(s) from project {tiles[0].project}, {total_mb:.1f} MB total")
    for tile in tiles:
        print(
            f"  {tile.filename}  {tile.size_bytes / 1e6:.1f} MB  published {tile.publication_date}"
        )
    return total_mb


def _fetch(region: RegionConfig, dest: Path, list_only: bool, yes: bool) -> list[Path] | None:
    tiles = select_tiles(search_tiles(region.bbox_wgs84), region.bbox_wgs84)
    total_mb = _print_tiles(tiles)
    if list_only:
        return None
    if total_mb > LARGE_DOWNLOAD_MB and not yes:
        print(f"Download exceeds {LARGE_DOWNLOAD_MB:.0f} MB; re-run with --yes to proceed.")
        return None
    return download_tiles(tiles, dest)


def _cmd_demo(args: argparse.Namespace) -> int:
    result = run_demo(
        args.out,
        seed=args.seed,
        scene_params=SceneParams(size_m=args.size_m),
        write_geotiffs=not args.no_geotiff,
    )
    stats = result.products.stats
    print(f"Synthetic scene: {stats.points_total:,} points, {stats.area_acres:.1f} ac")
    for area in result.products.class_table:
        print(f"  {area.name:<11} {area.acres:6.2f} ac ({area.percent:4.1f}%)")
    print(f"  treatment  {stats.treatment_acres:6.2f} ac")
    print("Validation against synthetic truth:")
    for check in result.checks:
        status = "pass" if check.passed else "FAIL"
        print(f"  [{status}] {check.name}: {check.value:.3f} {check.comparison} {check.limit:g}")
    print(f"Report: {result.outputs.report_md.as_posix()}")
    return EXIT_OK if all(c.passed for c in result.checks) else EXIT_CHECKS_FAILED


def _cmd_fetch(args: argparse.Namespace) -> int:
    _, region = _region_config(args.config)
    print(f"Region: {region.name} ({region.land_status})")
    for path in _fetch(region, args.dest, args.list_only, args.yes) or []:
        print(f"  saved {path}")
    return EXIT_OK


def _cmd_run(args: argparse.Namespace) -> int:
    if args.fetch:
        config, fetched_region = _region_config(args.config)
        region: RegionConfig | None = fetched_region
        laz = _fetch(fetched_region, args.dest, list_only=False, yes=args.yes)
        if laz is None:
            return EXIT_USAGE
    else:
        config = Config(None, PipelineParams()) if args.config is None else load_config(args.config)
        region = config.region
        laz = list(args.laz)
    out = args.out or DEFAULT_OUTPUT_DIR / _slug(region.name if region else "run")
    products, outputs = run_on_tiles(
        laz,
        out,
        config.pipeline,
        region=None if args.no_region_crop else region,
        zones_path=args.zones,
        crs=args.crs,
        z_units=cast(ZUnits, args.z_units),
        ground_method=cast(GroundMethod | None, args.ground),
        write_geotiffs=not args.no_geotiff,
    )
    stats = products.stats
    print(f"{stats.area_acres:.1f} ac processed; woody cover {stats.woody_acres:.2f} ac")
    print(f"Report: {outputs.report_md.as_posix()}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        0 on success, 1 if a demo truth check fails, 2 for usage, config or input errors,
        3 if a required optional dependency is missing.
    """
    args = build_parser().parse_args(argv)
    level = LOG_LEVELS[min(args.verbose, len(LOG_LEVELS) - 1)]
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    handlers = {"demo": _cmd_demo, "fetch": _cmd_fetch, "run": _cmd_run}
    try:
        return handlers[args.command](args)
    except LidarDependencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_MISSING_DEPENDENCY
    except LidarTerrainLabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
