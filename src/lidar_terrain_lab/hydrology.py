"""Surface-water routing: depression filling, D8 flow direction, accumulation, drainage lines.

Implemented directly on NumPy arrays: Priority-Flood+epsilon (Barnes, Lehman and Mulla,
2014, Computers & Geosciences 62) followed by steepest-descent D8 routing.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ._types import BoolArray, FloatArray, IntArray, UInt8Array
from .rasterize import Grid

# ESRI/TauDEM-style D8 codes, clockwise from east: E, SE, S, SW, W, NW, N, NE.
D8_CODES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
D8_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)
OUTLET = 0  # no downslope neighbor: the grid edge or a nodata boundary


def priority_flood_fill(dem: FloatArray) -> FloatArray:
    """Raise every closed depression to its spill point so all cells drain off the grid.

    Water can only leave a DEM across its edges. Starting from the edge cells, the
    algorithm always expands from the lowest cell reached so far (a priority queue). A
    newly reached neighbor that is lower than the cell it was reached from must sit in
    a pit, so it is raised to just above that cell. The "just above" (the next
    representable float) keeps a tiny downhill gradient across filled areas so that D8
    has a direction to follow everywhere. NaN cells are treated as outside the grid.

    Args:
        dem: Elevations; NaN marks nodata.

    Returns:
        The filled DEM (float64), identical to the input except inside depressions.
    """
    rows, cols = dem.shape
    valid = np.isfinite(dem)
    edge = np.zeros(dem.shape, dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    seeds = valid & (edge | ndimage.binary_dilation(~valid, structure=np.ones((3, 3))))

    elevation: list[float] = dem.astype(np.float64).ravel().tolist()
    closed = bytearray((~valid).ravel().astype(np.uint8).tobytes())
    open_heap: list[tuple[float, int]] = []
    for index in np.flatnonzero(seeds).tolist():
        open_heap.append((elevation[index], index))
        closed[index] = 1
    heapq.heapify(open_heap)
    pit: deque[int] = deque()

    while open_heap or pit:
        cell = pit.popleft() if pit else heapq.heappop(open_heap)[1]
        z_cell = elevation[cell]
        row, col = divmod(cell, cols)
        for d_row, d_col in D8_OFFSETS:
            n_row, n_col = row + d_row, col + d_col
            if not (0 <= n_row < rows and 0 <= n_col < cols):
                continue
            neighbor = n_row * cols + n_col
            if closed[neighbor]:
                continue
            closed[neighbor] = 1
            if elevation[neighbor] <= z_cell:
                elevation[neighbor] = math.nextafter(z_cell, math.inf)
                pit.append(neighbor)
            else:
                heapq.heappush(open_heap, (elevation[neighbor], neighbor))

    filled = np.asarray(elevation, dtype=np.float64).reshape(dem.shape)
    filled[~valid] = np.nan
    return filled


def d8_flow_direction(filled: FloatArray, res_m: float, edge_outlets: bool = True) -> UInt8Array:
    """Send each cell's water to its steepest downhill neighbor (one of eight).

    Drop is divided by the distance to the neighbor (1 cell or the diagonal, sqrt 2 cells).

    Args:
        filled: Depression-filled DEM.
        res_m: Cell size in meters.
        edge_outlets: Let every border cell drain off the grid. A tile is a window on a
            larger landscape; without this, border cells can only pass water sideways along
            the edge, which draws false channels on the boundary.

    Returns:
        A D8 code per cell (see ``D8_CODES``), ``OUTLET`` where water leaves the grid.
    """
    rows, cols = filled.shape
    padded = np.pad(filled, 1, constant_values=np.nan)
    best_drop = np.zeros(filled.shape)
    direction = np.full(filled.shape, OUTLET, dtype=np.uint8)
    for code, (d_row, d_col) in zip(D8_CODES, D8_OFFSETS, strict=True):
        neighbor = padded[1 + d_row : 1 + d_row + rows, 1 + d_col : 1 + d_col + cols]
        distance = res_m * (math.sqrt(2.0) if d_row and d_col else 1.0)
        drop = np.nan_to_num((filled - neighbor) / distance, nan=-np.inf)
        steeper = drop > best_drop
        best_drop = np.where(steeper, drop, best_drop)
        direction = np.where(steeper, np.uint8(code), direction)
    if edge_outlets:
        direction[[0, -1], :] = OUTLET
        direction[:, [0, -1]] = OUTLET
    return direction


def downstream_index(direction: UInt8Array) -> IntArray:
    """Flat index of the cell each cell drains into, -1 for outlets.

    Args:
        direction: D8 codes.

    Returns:
        One index per cell (flattened, row-major).
    """
    rows, cols = direction.shape
    row_idx, col_idx = np.indices(direction.shape)
    downstream = np.full(rows * cols, -1, dtype=np.int64)
    flat_direction = direction.ravel()
    for code, (d_row, d_col) in zip(D8_CODES, D8_OFFSETS, strict=True):
        cells = np.flatnonzero(flat_direction == code)
        target_rows = row_idx.ravel()[cells] + d_row
        target_cols = col_idx.ravel()[cells] + d_col
        downstream[cells] = target_rows * cols + target_cols
    return downstream


def flow_accumulation(direction: UInt8Array, weights: FloatArray | None = None) -> FloatArray:
    """How many cells (or how much weight) drain through each cell, itself included.

    Cells are processed in waves: first every cell nothing flows into (ridges), then each
    cell whose upstream neighbors are all done. Each wave is vectorized, so the loop runs
    once per cell of the longest flow path rather than once per cell of the grid.

    Args:
        direction: D8 codes.
        weights: Optional per-cell contribution (defaults to 1 per cell).

    Returns:
        Accumulated upstream weight per cell.
    """
    size = direction.size
    downstream = downstream_index(direction)
    total = (
        np.ones(size, dtype=np.float64)
        if weights is None
        else weights.astype(np.float64).ravel().copy()
    )
    receives = downstream >= 0
    inflow_count = np.bincount(downstream[receives], minlength=size)
    frontier = np.flatnonzero(inflow_count == 0)
    while frontier.size:
        targets = downstream[frontier]
        flows = targets >= 0
        sources, targets = frontier[flows], targets[flows]
        np.add.at(total, targets, total[sources])
        np.subtract.at(inflow_count, targets, 1)
        candidates = np.unique(targets)
        frontier = candidates[inflow_count[candidates] == 0]
    return total.reshape(direction.shape)


@dataclass(frozen=True)
class DrainageSegment:
    """One reach of the drainage network between junctions.

    Attributes:
        cells: Flat cell indices from upstream to downstream.
        xy: Cell-center coordinates in CRS units, shape ``(n, 2)``.
        strahler_order: 1 for headwater reaches; two joining order-n reaches make order n+1.
        length_m: Path length along cell centers in meters.
        contributing_area_ha: Area draining to the reach's downstream end.
    """

    cells: tuple[int, ...]
    xy: FloatArray
    strahler_order: int
    length_m: float
    contributing_area_ha: float


def strahler_orders(
    stream_cells: IntArray, downstream: IntArray, accumulation: FloatArray
) -> IntArray:
    """Strahler order of each stream cell.

    Accumulation strictly grows downstream, so sorting by it gives an upstream-first order.

    Args:
        stream_cells: Flat indices of stream cells.
        downstream: Flat downstream index per cell (-1 for outlets).
        accumulation: Accumulated cell count per cell (flattened).

    Returns:
        Order per cell of the full grid (0 off the network).
    """
    order = np.zeros(downstream.size, dtype=np.int64)
    best_in = np.zeros(downstream.size, dtype=np.int64)
    count_best = np.zeros(downstream.size, dtype=np.int64)
    is_stream = np.zeros(downstream.size, dtype=bool)
    is_stream[stream_cells] = True
    for cell in stream_cells[np.argsort(accumulation[stream_cells], kind="stable")].tolist():
        incoming = best_in[cell]
        cell_order = 1 if incoming == 0 else incoming + (1 if count_best[cell] >= 2 else 0)
        order[cell] = cell_order
        target = downstream[cell]
        if target >= 0 and is_stream[target]:
            if cell_order > best_in[target]:
                best_in[target] = cell_order
                count_best[target] = 1
            elif cell_order == best_in[target]:
                count_best[target] += 1
    return order


def drainage_network(
    direction: UInt8Array, accumulation: FloatArray, grid: Grid, threshold_cells: float
) -> tuple[BoolArray, IntArray, list[DrainageSegment]]:
    """Extract drainage lines where accumulated flow passes a threshold.

    Args:
        direction: D8 codes.
        accumulation: Accumulated cell counts.
        grid: Grid for coordinates and cell size.
        threshold_cells: Minimum number of contributing cells for a channel.

    Returns:
        Channel mask, Strahler order raster, and the reaches as polylines.
    """
    downstream = downstream_index(direction)
    flat_acc = accumulation.ravel()
    is_stream = flat_acc >= threshold_cells
    stream_cells = np.flatnonzero(is_stream)
    order = strahler_orders(stream_cells, downstream, flat_acc)

    # Accumulation only grows downstream, so a channel cell always drains into a channel cell.
    stream_down = np.where(is_stream, downstream, -1)
    upstream_count = np.bincount(stream_down[stream_down >= 0], minlength=downstream.size)
    starts = stream_cells[(upstream_count[stream_cells] == 0) | (upstream_count[stream_cells] >= 2)]

    xs, ys = grid.cell_centers()
    cell_area_ha = grid.cell_area_m2 / 10_000.0
    segments: list[DrainageSegment] = []
    for start in starts.tolist():
        path = [start]
        cell = start
        while True:
            nxt = int(stream_down[cell])
            if nxt < 0:
                break
            path.append(nxt)
            if upstream_count[nxt] >= 2:
                break
            cell = nxt
        if len(path) < 2:
            continue
        cells = np.asarray(path, dtype=np.int64)
        row, col = np.divmod(cells, grid.width)
        xy = np.column_stack([xs[col], ys[row]])
        steps = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])) * grid.meters_per_unit
        end = path[-2] if upstream_count[path[-1]] >= 2 else path[-1]
        segments.append(
            DrainageSegment(
                cells=tuple(path),
                xy=xy,
                strahler_order=int(order[start]),
                length_m=float(steps.sum()),
                contributing_area_ha=float(flat_acc[end] * cell_area_ha),
            )
        )
    return is_stream.reshape(direction.shape), order.reshape(direction.shape), segments


def simplify_staircase(xy: FloatArray) -> FloatArray:
    """Drop interior vertices where a D8 path keeps going in the same direction.

    Args:
        xy: Vertices of a polyline, shape ``(n, 2)``.

    Returns:
        The same line with collinear interior vertices removed.
    """
    if len(xy) <= 2:
        return xy
    steps = np.sign(np.diff(xy, axis=0))
    turns = np.any(steps[1:] != steps[:-1], axis=1)
    keep: BoolArray = np.concatenate([[True], turns, [True]])
    return xy[keep]
