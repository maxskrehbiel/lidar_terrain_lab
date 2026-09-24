"""Pipeline parameters and the optional study-region definition, loaded from TOML.

Every tunable number lives in a frozen dataclass here rather than inside the algorithms.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, TypeVar

from .errors import ConfigError

GroundMethod = Literal["auto", "existing", "pmf"]
DEMO_REGION_RESOURCE = "demo_region.toml"


@dataclass(frozen=True)
class GroundParams:
    """Settings for separating ground returns from everything above them.

    Attributes:
        method: ``existing`` trusts the provider's class 2, ``pmf`` runs the morphological
            filter, ``auto`` uses the provider's classes when enough points carry class 2.
        min_existing_fraction: Share of class-2 points that ``auto`` needs to trust them.
        max_window_m: Largest opening window; must exceed the widest crown or roof.
        terrain_slope: Expected steepest terrain slope (rise over run) for the filter.
        initial_threshold_m: Height tolerance at the smallest window.
        max_threshold_m: Cap on the height tolerance at large windows.
        tolerance_m: A point within this distance of the ground surface is ground.
    """

    method: GroundMethod = "auto"
    min_existing_fraction: float = 0.05
    max_window_m: float = 20.0
    terrain_slope: float = 0.3
    initial_threshold_m: float = 0.3
    max_threshold_m: float = 3.0
    tolerance_m: float = 0.25


@dataclass(frozen=True)
class CanopyParams:
    """Settings for the canopy height model.

    Attributes:
        max_height_m: Heights above this are treated as unflagged noise.
        pit_fill_m: A cell this far below its 3x3 median is a pit and takes the median.
    """

    max_height_m: float = 45.0
    pit_fill_m: float = 1.5


@dataclass(frozen=True)
class HydrologyParams:
    """Settings for drainage extraction.

    Attributes:
        channel_threshold_ha: Contributing area at which a cell counts as a drainage line.
        sink_report_depth_m: Filled depressions deeper than this are reported as sinks.
    """

    channel_threshold_ha: float = 1.0
    sink_report_depth_m: float = 0.1


@dataclass(frozen=True)
class VegetationParams:
    """Canopy-height classes and which of them count toward treatment acreage.

    Attributes:
        breaks_m: Class boundaries in meters; ``n`` breaks make ``n + 1`` classes.
        names: One name per class, lowest first.
        colors: One hex color per class for maps and overlays.
        woody: Class names that count as woody cover (drawn in color on maps).
        treatment: Class names whose area is summed as candidate treatment acreage.
    """

    breaks_m: tuple[float, ...] = (0.5, 2.0, 5.0)
    names: tuple[str, ...] = ("Open", "Shrub", "Small tree", "Tree")
    colors: tuple[str, ...] = ("#f1efe6", "#8cbb5c", "#4a8f3f", "#1c552d")
    woody: tuple[str, ...] = ("Shrub", "Small tree", "Tree")
    treatment: tuple[str, ...] = ("Shrub", "Small tree")


@dataclass(frozen=True)
class HillshadeParams:
    """Sun position and vertical exaggeration for shaded relief.

    Attributes:
        azimuth_deg: Compass direction the light comes from, clockwise from north.
        altitude_deg: Sun height above the horizon.
        z_factor: Multiplier applied to elevations before shading.
    """

    azimuth_deg: float = 315.0
    altitude_deg: float = 45.0
    z_factor: float = 1.0


@dataclass(frozen=True)
class PipelineParams:
    """All pipeline settings.

    Attributes:
        resolution_m: Raster cell size in meters.
        ground: Ground filter settings.
        canopy: Canopy height model settings.
        hydrology: Drainage settings.
        vegetation: Height classes.
        hillshade: Shaded relief settings.
    """

    resolution_m: float = 1.0
    ground: GroundParams = field(default_factory=GroundParams)
    canopy: CanopyParams = field(default_factory=CanopyParams)
    hydrology: HydrologyParams = field(default_factory=HydrologyParams)
    vegetation: VegetationParams = field(default_factory=VegetationParams)
    hillshade: HillshadeParams = field(default_factory=HillshadeParams)


@dataclass(frozen=True)
class RegionConfig:
    """A study area on public land, used to fetch and crop real 3DEP tiles.

    Attributes:
        name: Display name.
        land_status: Who manages the land (must be public for anything shared).
        bbox_wgs84: ``(west, south, east, north)`` in decimal degrees.
        notes: Free text shown in reports.
    """

    name: str
    land_status: str
    bbox_wgs84: tuple[float, float, float, float]
    notes: str = ""


@dataclass(frozen=True)
class Config:
    """A parsed configuration file."""

    region: RegionConfig | None
    pipeline: PipelineParams


_T = TypeVar("_T")


def _build(cls: type[_T], table: dict[str, Any], where: str) -> _T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    known = {f.name: f for f in fields(cls)}
    unknown = set(table) - set(known)
    if unknown:
        raise ConfigError(f"unknown key(s) in [{where}]: {', '.join(sorted(unknown))}")
    kwargs: dict[str, Any] = {}
    for name, value in table.items():
        default = known[name].default_factory
        if isinstance(value, dict) and callable(default):
            kwargs[name] = _build(type(default()), value, f"{where}.{name}")
        elif isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    try:
        return cls(**kwargs)
    except TypeError as exc:  # a required key is missing
        raise ConfigError(f"[{where}]: {exc}") from exc


def load_config(path: str | Path) -> Config:
    """Load a TOML configuration file.

    Args:
        path: File with optional ``[region]`` and ``[pipeline]`` tables.

    Returns:
        The parsed configuration; missing values take their defaults.

    Raises:
        ConfigError: If the file is missing or unreadable, has unknown keys, or an
            invalid bounding box.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc.strerror or exc}") from exc
    return parse_config(text, str(path))


def load_demo_region_config() -> Config:
    """Load the example region that ships with the package (a box on public land).

    Returns:
        The packaged ``demo_region.toml`` configuration.
    """
    resource = files("lidar_terrain_lab").joinpath("data", DEMO_REGION_RESOURCE)
    return parse_config(resource.read_text(encoding="utf-8"), DEMO_REGION_RESOURCE)


def parse_config(text: str, source: str = "<string>") -> Config:
    """Parse TOML configuration text.

    Args:
        text: TOML with optional ``[region]`` and ``[pipeline]`` tables.
        source: Name used in error messages.

    Returns:
        The parsed configuration.

    Raises:
        ConfigError: For invalid TOML, unknown keys or an invalid bounding box.
    """
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{source} is not valid TOML: {exc}") from exc
    region = None
    if "region" in data:
        region = _build(RegionConfig, data["region"], "region")
        box = region.bbox_wgs84
        valid = len(box) == 4 and -180 <= box[0] < box[2] <= 180 and -90 <= box[1] < box[3] <= 90
        if not valid:
            raise ConfigError(f"invalid bbox_wgs84 {box}; expected [west, south, east, north]")
    pipeline = _build(PipelineParams, data.get("pipeline", {}), "pipeline")
    return Config(region=region, pipeline=pipeline)
