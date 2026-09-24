"""Depression filling, D8 routing, accumulation and the drainage network."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import inclined_plane

from lidar_terrain_lab.hydrology import (
    D8_CODES,
    OUTLET,
    d8_flow_direction,
    downstream_index,
    drainage_network,
    flow_accumulation,
    priority_flood_fill,
    simplify_staircase,
    strahler_orders,
)
from lidar_terrain_lab.pipeline import Products
from lidar_terrain_lab.rasterize import Grid
from lidar_terrain_lab.synthetic import SyntheticScene

EAST = D8_CODES[0]


def _v_valley(rows: int = 21, cols: int = 30) -> np.ndarray:
    """Sides slope toward a channel on the middle row, which falls toward the east."""
    row, col = np.indices((rows, cols), dtype=np.float64)
    return 50.0 + 0.5 * np.abs(row - rows // 2) - 0.05 * col


def _every_cell_drains(filled: np.ndarray) -> bool:
    """True if each interior cell has a strictly lower neighbor."""
    rows, cols = filled.shape
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if not (filled[r - 1 : r + 2, c - 1 : c + 2] < filled[r, c]).any():
                return False
    return True


def test_fill_raises_a_pit_to_its_spill_point() -> None:
    dem = inclined_plane(15, 15, 0.1, 0.0)
    dem[5:8, 5:8] -= 3.0
    filled = priority_flood_fill(dem)
    spill = dem[6, 4]  # lowest rim cell, on the downhill (west) side
    assert filled[6, 6] >= spill
    assert filled[6, 6] - spill < 1e-9
    outside = np.ones(dem.shape, dtype=bool)
    outside[4:9, 4:9] = False
    assert np.array_equal(filled[outside], dem[outside])
    assert _every_cell_drains(filled)


def test_fill_treats_nodata_as_an_outlet() -> None:
    dem = np.full((9, 9), 10.0)
    dem[4, 4] = np.nan
    dem[3:6, 3] = 5.0  # low cells beside the nodata hole drain into it
    dem[7, 7] = 5.0  # a true pit elsewhere
    filled = priority_flood_fill(dem)
    assert np.isnan(filled[4, 4])
    assert (filled[3:6, 3] == 5.0).all()
    assert filled[7, 7] > 10.0


def test_d8_on_an_eastward_plane() -> None:
    direction = d8_flow_direction(inclined_plane(8, 8, -0.1, 0.0), res_m=1.0)
    assert (direction[1:-1, 1:-1] == EAST).all()
    assert (direction[0, :] == OUTLET).all()
    inward = d8_flow_direction(inclined_plane(8, 8, -0.1, 0.0), 1.0, edge_outlets=False)
    assert inward[0, 0] == EAST
    assert inward[3, -1] == OUTLET  # nothing lower inside the grid


def test_downstream_index_follows_codes() -> None:
    direction = np.array([[EAST, OUTLET], [D8_CODES[6], OUTLET]], dtype=np.uint8)
    assert downstream_index(direction).tolist() == [1, -1, 0, -1]


def test_accumulation_conserves_cells_and_concentrates_in_the_valley() -> None:
    dem = _v_valley()
    direction = d8_flow_direction(priority_flood_fill(dem), res_m=1.0)
    accumulation = flow_accumulation(direction)
    outlets = direction == OUTLET
    assert accumulation[outlets].sum() == pytest.approx(dem.size)
    rows, cols = dem.shape
    mid = rows // 2
    assert np.argmax(accumulation[:, -2]) == mid
    # Every interior cell drains to the valley row and leaves through its east end.
    assert accumulation[mid, -2] >= (rows - 2) * (cols - 2)


def test_weighted_accumulation() -> None:
    direction = d8_flow_direction(inclined_plane(3, 6, -0.1, 0.0), 1.0, edge_outlets=False)
    weights = np.full((3, 6), 2.0)
    assert flow_accumulation(direction, weights)[1, -1] == pytest.approx(12.0)


def test_strahler_order_on_a_hand_built_network() -> None:
    # Cells 0 and 1 join at 2, which drains to 3; cell 4 joins at 3.
    downstream = np.array([2, 2, 3, -1, 3])
    accumulation = np.array([1.0, 1.0, 3.0, 5.0, 1.0])
    order = strahler_orders(np.arange(5), downstream, accumulation)
    assert order.tolist() == [1, 1, 2, 2, 1]


def test_drainage_network_on_a_valley() -> None:
    dem = _v_valley()
    grid = Grid(0.0, float(dem.shape[0]), 1.0, dem.shape[1], dem.shape[0], "EPSG:32631")
    direction = d8_flow_direction(priority_flood_fill(dem), res_m=1.0)
    accumulation = flow_accumulation(direction)
    channels, order, segments = drainage_network(direction, accumulation, grid, 60.0)
    assert channels[dem.shape[0] // 2, -5:].all()
    assert order.max() >= 1
    assert segments
    main = max(segments, key=lambda s: s.contributing_area_ha)
    assert main.length_m > 10
    assert main.contributing_area_ha == pytest.approx(accumulation.max() * 1e-4, rel=0.05)


def test_scene_drainage_follows_the_true_channel(scene: SyntheticScene, products: Products) -> None:
    size = scene.params.size_m
    true_v = np.interp(np.arange(size) + 0.5, scene.channel_uv[:, 0], scene.channel_uv[:, 1])
    offsets = []
    for col in range(int(0.4 * size), int(0.95 * size)):
        rows = np.flatnonzero(products.hydrology.channel_mask[:, col])
        near = rows[np.abs((size - rows - 0.5) - true_v[col]) < 20]
        assert near.size, f"no drainage line near the channel at column {col}"
        offsets.append(np.min(np.abs((size - near - 0.5) - true_v[col])))
    assert np.median(offsets) <= 1.5
    assert products.stats.max_strahler_order >= 2


def test_simplify_staircase_drops_collinear_vertices() -> None:
    xy = np.array([[0, 0], [1, 0], [2, 0], [3, 1], [4, 2], [4, 3]], dtype=float)
    assert simplify_staircase(xy).tolist() == [[0, 0], [2, 0], [4, 2], [4, 3]]
    assert simplify_staircase(xy[:2]).shape == (2, 2)
