"""Colors, ramps and PNG settings shared by the map plates and the Google Earth overlays.

Magnitudes use one hue from light to dark; the lightest step stays visible over relief.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
DRAINAGE_COLOR = "#2a78d6"
ZONE_COLOR = "#d95926"
ELEVATION_RAMP = ("#f4ecdf", "#c9a77c", "#7a5230")
CANOPY_RAMP = ("#d4e9bf", "#8cbb5c", "#4a8f3f", "#1c552d")
FLOW_RAMP = ("#b7d3f6", "#5598e7", "#1c5cab", "#0d366b")
SLOPE_COLORS = ("#fde3cc", "#f6b27a", "#e0772f", "#a3440f")
MIN_MAPPED_CONTRIBUTING_AREA_M2 = 200.0
# Dropping matplotlib's "Software" tag keeps PNGs byte-identical across versions and re-runs.
PNG_METADATA: dict[str, str | None] = {"Software": None}


def ramp(name: str, colors: tuple[str, ...]) -> LinearSegmentedColormap:
    """Build a continuous colormap from an ordered tuple of hex colors."""
    return LinearSegmentedColormap.from_list(name, list(colors))
