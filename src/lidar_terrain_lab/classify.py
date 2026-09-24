"""Vegetation-height classes from the canopy height model, and acreage per class."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from ._types import BoolArray, FloatArray, UInt8Array
from .config import VegetationParams
from .errors import ConfigError
from .rasterize import SQUARE_METERS_PER_ACRE, SQUARE_METERS_PER_HECTARE

NODATA_CLASS = 0


@dataclass(frozen=True)
class VegClass:
    """One canopy-height class.

    Attributes:
        code: Raster value (1-based; 0 is nodata).
        name: Display name.
        min_m: Inclusive lower height bound in meters.
        max_m: Exclusive upper bound in meters (``inf`` for the top class).
        color: Hex color for maps.
        woody: Whether the class counts as woody cover.
        treatment: Whether the class counts toward treatment acreage.
    """

    code: int
    name: str
    min_m: float
    max_m: float
    color: str
    woody: bool
    treatment: bool

    @property
    def label(self) -> str:
        """Name with its height range, e.g. ``Shrub (0.5-2 m)``."""
        if math.isinf(self.max_m):
            return f"{self.name} (>{self.min_m:g} m)"
        if self.min_m <= 0:
            return f"{self.name} (<{self.max_m:g} m)"
        return f"{self.name} ({self.min_m:g}-{self.max_m:g} m)"


@dataclass(frozen=True)
class ClassArea:
    """Area of one class inside some mask.

    Attributes:
        name: Class name.
        cells: Number of cells.
        acres: Area in acres.
        hectares: Area in hectares.
        percent: Share of the masked area.
        woody: Whether the class counts as woody cover.
        treatment: Whether the class counts toward treatment acreage.
    """

    name: str
    cells: int
    acres: float
    hectares: float
    percent: float
    woody: bool
    treatment: bool


def build_classes(params: VegetationParams) -> list[VegClass]:
    """Turn height breaks into class definitions.

    Args:
        params: Breaks, names, colors and the woody and treatment class names.

    Returns:
        Classes ordered from lowest to tallest.

    Raises:
        ConfigError: If the counts disagree, the breaks do not increase, or a woody or
            treatment class name is not defined.
    """
    n_classes = len(params.breaks_m) + 1
    if len(params.names) != n_classes or len(params.colors) != n_classes:
        raise ConfigError(f"{len(params.breaks_m)} breaks need {n_classes} names and colors")
    if any(b <= a for a, b in pairwise(params.breaks_m)) or min(params.breaks_m, default=1) <= 0:
        raise ConfigError(f"class breaks must be positive and increasing: {params.breaks_m}")
    for role, names in (("woody", params.woody), ("treatment", params.treatment)):
        unknown = set(names) - set(params.names)
        if unknown:
            raise ConfigError(f"{role} classes not defined: {sorted(unknown)}")
    edges = (0.0, *params.breaks_m, math.inf)
    return [
        VegClass(
            code=i + 1,
            name=name,
            min_m=edges[i],
            max_m=edges[i + 1],
            color=color,
            woody=name in params.woody,
            treatment=name in params.treatment,
        )
        for i, (name, color) in enumerate(zip(params.names, params.colors, strict=True))
    ]


def classify_heights(chm: FloatArray, classes: list[VegClass]) -> UInt8Array:
    """Assign each cell the code of the class its canopy height falls in.

    Args:
        chm: Canopy height in meters (NaN is nodata).
        classes: Output of :func:`build_classes`, lowest class first.

    Returns:
        Class codes, ``NODATA_CLASS`` where the height is NaN.
    """
    breaks = np.array([c.min_m for c in classes[1:]])
    codes = (np.digitize(np.nan_to_num(chm, nan=0.0), breaks) + 1).astype(np.uint8)
    codes[~np.isfinite(chm)] = NODATA_CLASS
    return codes


def class_areas(
    codes: UInt8Array,
    classes: list[VegClass],
    cell_area_m2: float,
    mask: BoolArray | None = None,
) -> list[ClassArea]:
    """Tabulate the area of each class, optionally inside a mask.

    Args:
        codes: Class raster.
        classes: Class definitions.
        cell_area_m2: Area of one cell.
        mask: Cells to include (defaults to every cell with data).

    Returns:
        One row per class in class order.
    """
    inside = codes != NODATA_CLASS if mask is None else mask & (codes != NODATA_CLASS)
    total_cells = int(inside.sum())
    rows = []
    for veg in classes:
        cells = int((inside & (codes == veg.code)).sum())
        area = cells * cell_area_m2
        rows.append(
            ClassArea(
                name=veg.name,
                cells=cells,
                acres=area / SQUARE_METERS_PER_ACRE,
                hectares=area / SQUARE_METERS_PER_HECTARE,
                percent=100.0 * cells / total_cells if total_cells else 0.0,
                woody=veg.woody,
                treatment=veg.treatment,
            )
        )
    return rows


def woody_floor_m(classes: list[VegClass]) -> float:
    """Lowest height at which a class counts as woody.

    Args:
        classes: Class definitions.

    Returns:
        The smallest ``min_m`` among woody classes (0 if none are woody).
    """
    return min((c.min_m for c in classes if c.woody), default=0.0)


def woody_acres(areas: list[ClassArea]) -> float:
    """Sum the acreage of the woody classes.

    Args:
        areas: Output of :func:`class_areas`.

    Returns:
        Acres.
    """
    return sum(a.acres for a in areas if a.woody)


def treatment_acres(areas: list[ClassArea]) -> float:
    """Sum the acreage of the treatment classes.

    Args:
        areas: Output of :func:`class_areas`.

    Returns:
        Acres.
    """
    return sum(a.acres for a in areas if a.treatment)
