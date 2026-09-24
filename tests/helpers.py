"""Small builders shared by several test modules."""

from __future__ import annotations

import numpy as np

from lidar_terrain_lab._types import FloatArray


def inclined_plane(rows: int, cols: int, dz_dx: float, dz_dy: float) -> FloatArray:
    """Elevations of a plane rising ``dz_dx`` per cell eastward and ``dz_dy`` northward."""
    row, col = np.indices((rows, cols), dtype=np.float64)
    return 100.0 + dz_dx * col + dz_dy * (rows - 1 - row)
