"""Seeded synthetic LiDAR scene with known ground, canopy and channel truth.

The scene is georeferenced just north-east of 0 N, 0 E (open ocean in the Gulf of Guinea,
the conventional "Null Island" placeholder) so its outputs cannot be mistaken for real land.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from rasterio.warp import transform as transform_points
from scipy.spatial import cKDTree

from ._types import BoolArray, FloatArray, IntArray, UInt8Array
from .errors import ConfigError
from .point_cloud import PointCloud
from .rasterize import Grid

SYNTHETIC_CRS = "EPSG:32631"  # WGS 84 / UTM zone 31N
SYNTHETIC_ORIGIN_XY = (166_600.0, 1_000.0)  # about 0.005 E, 0.009 N
UNCLASSIFIED = 1
LOW_NOISE = 7
HIGH_NOISE = 18
SHRUB, SMALL_TREE, TREE = 0, 1, 2
TRIBUTARY_REACH_M = 100.0
MIN_SCENE_SIZE_M = 60.0  # smaller scenes leave no room for plants away from the channels
MAX_PLACEMENT_ROUNDS = 50


@dataclass(frozen=True)
class SceneParams:
    """Shape and sampling of the synthetic landscape.

    Attributes:
        size_m: Side of the square scene in meters.
        res_m: Raster cell size the truth rasters are built for.
        base_elevation_m: Elevation of the valley floor at the west edge.
        valley_gradient: Down-valley fall toward the east (rise over run).
        side_slope: Slope of the valley sides toward the channel.
        channel_depth_m: Incision depth of the main channel.
        channel_width_m: Gaussian half-width of the main channel.
        tributary_depth_m: Incision depth of the tributary.
        tributary_width_m: Gaussian half-width of the tributary.
        hill_amplitude_m: Amplitude of the rolling upland relief.
        ground_density: Ground returns per square meter in the open.
        canopy_top_density: Returns per square meter on crown surfaces.
        canopy_interior_density: Returns per square meter inside crowns.
        ground_keep_under_canopy: Share of ground returns that penetrate a crown.
        ground_noise_m: Standard deviation of ground-return height error.
        trees_per_ha: Mean number of trees (6-12 m) per hectare.
        small_trees_per_ha: Mean number of small trees (2.4-4.6 m) per hectare.
        shrub_clusters_per_ha: Mean number of shrub patches per hectare.
        shrubs_per_cluster: Mean shrubs (0.8-1.8 m) per patch.
        noise_points: Provider-flagged noise points of each kind (low and high).
        truth_supersample: Sub-cells per cell side used to build the canopy truth.
    """

    size_m: float = 400.0
    res_m: float = 1.0
    base_elevation_m: float = 320.0
    valley_gradient: float = 0.012
    side_slope: float = 0.06
    channel_depth_m: float = 2.5
    channel_width_m: float = 6.0
    tributary_depth_m: float = 1.5
    tributary_width_m: float = 4.0
    hill_amplitude_m: float = 1.6
    ground_density: float = 6.0
    canopy_top_density: float = 8.0
    canopy_interior_density: float = 3.0
    ground_keep_under_canopy: float = 0.3
    ground_noise_m: float = 0.03
    trees_per_ha: float = 6.0
    small_trees_per_ha: float = 30.0
    shrub_clusters_per_ha: float = 4.0
    shrubs_per_cluster: float = 25.0
    noise_points: int = 25
    truth_supersample: int = 4


@dataclass(frozen=True)
class Plants:
    """Crown positions and sizes in local meters (east, north of the south-west corner).

    Attributes:
        u: Crown center east coordinate.
        v: Crown center north coordinate.
        height: Crown top height above ground in meters.
        radius: Crown radius in meters.
        kind: ``SHRUB``, ``SMALL_TREE`` or ``TREE``.
    """

    u: FloatArray
    v: FloatArray
    height: FloatArray
    radius: FloatArray
    kind: IntArray

    def __len__(self) -> int:
        return len(self.u)


@dataclass(frozen=True)
class SyntheticScene:
    """A synthetic point cloud and the truth it was generated from.

    Attributes:
        cloud: The points, as a sensor would deliver them (ground not yet classified).
        grid: The raster grid the truth rasters use.
        params: Generation settings.
        seed: Random seed.
        is_ground: True for every point generated on the ground surface.
        height_above_ground: True height of each point above the ground surface.
        ground_true: Ground elevation at each cell center.
        chm_true: Highest crown surface within each cell (what a perfect max-return
            canopy model at this resolution would show).
        plants: Every crown.
        channel_uv: Main-channel centerline in local meters, shape ``(n, 2)``.
        zones: Three management units as a GeoJSON FeatureCollection in longitude/latitude.
    """

    cloud: PointCloud
    grid: Grid
    params: SceneParams
    seed: int
    is_ground: BoolArray
    height_above_ground: FloatArray
    ground_true: FloatArray
    chm_true: FloatArray
    plants: Plants
    channel_uv: FloatArray
    zones: dict[str, Any]


class _WaveField:
    """A smooth random surface: a sum of plane waves with random directions and phases.

    Unlike a filtered noise raster, it can be evaluated exactly at any point, which the
    truth comparisons rely on.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        count: int,
        wavelength_range_m: tuple[float, float],
        amplitude_m: float,
    ) -> None:
        wavelengths = rng.uniform(*wavelength_range_m, count)
        angles = rng.uniform(0.0, 2.0 * math.pi, count)
        self.k_u = 2.0 * math.pi * np.cos(angles) / wavelengths
        self.k_v = 2.0 * math.pi * np.sin(angles) / wavelengths
        self.phase = rng.uniform(0.0, 2.0 * math.pi, count)
        # Longer waves get larger amplitudes; scaled so the field's RMS is about amplitude_m.
        weights = wavelengths / wavelengths.mean()
        self.amplitude = amplitude_m * math.sqrt(2.0) * weights / np.sqrt(np.sum(weights**2))

    def __call__(self, u: FloatArray, v: FloatArray) -> FloatArray:
        total = np.zeros(np.broadcast(u, v).shape)
        for ku, kv, ph, amp in zip(self.k_u, self.k_v, self.phase, self.amplitude, strict=True):
            total += amp * np.cos(ku * u + kv * v + ph)
        return total


class SyntheticTerrain:
    """Analytic ground: a meandering incised channel in a tilted valley with rolling uplands.

    The channel enters at the west edge and leaves at the east edge; a curved tributary
    drains the north side and joins it. Upland relief is damped near both channels so the
    valley floor falls steadily and the channel position is unambiguous.
    """

    def __init__(self, params: SceneParams, rng: np.random.Generator) -> None:
        """Draw the random shapes that make each seed's landscape distinct."""
        self.params = params
        size = params.size_m
        self.bend_phase = rng.uniform(0.0, 2.0 * math.pi, size=2)
        self.hills = _WaveField(rng, 12, (0.3 * size, 1.2 * size), params.hill_amplitude_m)
        self.texture = _WaveField(rng, 10, (12.0, 40.0), 0.06 * params.hill_amplitude_m)
        junction_u = 0.62 * size
        junction_v = float(self.channel_v(np.array([junction_u]))[0])
        start = np.array([0.48 * size, size])
        bend = np.array([0.66 * size, 0.5 * (size + junction_v) + rng.uniform(-0.05, 0.05) * size])
        end = np.array([junction_u, junction_v])
        t = np.linspace(0.0, 1.0, 600)[:, None]
        # Quadratic Bezier curve from the north edge to the junction.
        self.tributary_uv = (1 - t) ** 2 * start + 2 * (1 - t) * t * bend + t**2 * end
        self._tributary_index = cKDTree(self.tributary_uv)

    def channel_v(self, u: FloatArray) -> FloatArray:
        """North coordinate of the main-channel centerline at each east coordinate."""
        size = self.params.size_m
        long_bend = 0.10 * size * np.sin(2 * np.pi * u / (0.9 * size) + self.bend_phase[0])
        short_bend = 0.035 * size * np.sin(2 * np.pi * u / (0.33 * size) + self.bend_phase[1])
        return np.asarray(0.5 * size + long_bend + short_bend, dtype=np.float64)

    def tributary_distance(self, u: FloatArray, v: FloatArray) -> FloatArray:
        """Distance from each point to the tributary centerline, capped at ``TRIBUTARY_REACH_M``.

        Every term that uses this distance has decayed to nothing well before the cap, and
        bounding the search keeps the nearest-neighbor query fast for distant points.
        """
        points = np.column_stack([np.ravel(u), np.ravel(v)])
        distance, _ = self._tributary_index.query(
            points, distance_upper_bound=TRIBUTARY_REACH_M, workers=-1
        )
        capped = np.minimum(np.asarray(distance, dtype=np.float64), TRIBUTARY_REACH_M)
        return capped.reshape(np.shape(u))

    def elevation(self, u: FloatArray, v: FloatArray) -> FloatArray:
        """Ground elevation in meters at local coordinates."""
        p = self.params
        offset = v - self.channel_v(u)
        to_tributary = self.tributary_distance(u, v)
        # A hyperbola gives V-shaped valley sides with a rounded floor.
        valley = (
            p.base_elevation_m
            - p.valley_gradient * u
            + p.side_slope * (np.sqrt(offset**2 + 15.0**2) - 15.0)
        )
        # Taking the deeper of the two incisions avoids a spurious pit at the junction.
        incision = np.maximum(
            p.channel_depth_m * np.exp(-((offset / p.channel_width_m) ** 2)),
            p.tributary_depth_m * np.exp(-((to_tributary / p.tributary_width_m) ** 2)),
        )
        damping = (1.0 - np.exp(-((offset / 35.0) ** 2))) * (
            1.0 - np.exp(-((to_tributary / 20.0) ** 2))
        )
        relief = damping * (self.hills(u, v) + self.texture(u, v))
        return np.asarray(valley - incision + relief, dtype=np.float64)


def crown_height(du: FloatArray, dv: FloatArray, height: float, radius: float) -> FloatArray:
    """Dome-shaped crown surface: ``h * sqrt(1 - (r / R)**2)`` inside the crown, else 0."""
    r2 = (du**2 + dv**2) / radius**2
    return np.where(r2 < 1.0, height * np.sqrt(np.clip(1.0 - r2, 0.0, None)), 0.0)


def _positions(
    rng: np.random.Generator,
    terrain: SyntheticTerrain,
    count: int,
    min_channel_m: float,
    min_tributary_m: float,
) -> tuple[FloatArray, FloatArray]:
    """Uniform positions at least the given distances from both channels."""
    size = terrain.params.size_m
    us: list[float] = []
    vs: list[float] = []
    for _ in range(MAX_PLACEMENT_ROUNDS):
        if len(us) >= count:
            break
        u = rng.uniform(0.0, size, 4 * count + 16)
        v = rng.uniform(0.0, size, 4 * count + 16)
        ok = (np.abs(v - terrain.channel_v(u)) > min_channel_m) & (
            terrain.tributary_distance(u, v) > min_tributary_m
        )
        us.extend(u[ok].tolist())
        vs.extend(v[ok].tolist())
    if len(us) < count:
        raise ConfigError(f"a {size:g} m scene has no room for {count} plants away from channels")
    return np.asarray(us[:count]), np.asarray(vs[:count])


def _small_tree_positions(
    rng: np.random.Generator, terrain: SyntheticTerrain, count: int
) -> tuple[FloatArray, FloatArray]:
    """Half on the channel banks, where woody encroachment often starts; half anywhere."""
    size = terrain.params.size_m
    n_bank = count // 2
    bank_u = rng.uniform(0.0, size, n_bank)
    bank_side = rng.choice([-1.0, 1.0], n_bank)
    bank_v = np.clip(
        terrain.channel_v(bank_u) + bank_side * rng.uniform(8.0, 45.0, n_bank), 0, size
    )
    far_u, far_v = _positions(rng, terrain, count - n_bank, 8.0, 6.0)
    return np.concatenate([bank_u, far_u]), np.concatenate([bank_v, far_v])


def _shrub_positions(
    rng: np.random.Generator, terrain: SyntheticTerrain, n_clusters: int
) -> tuple[FloatArray, FloatArray]:
    """Shrubs scattered around patch centers, kept inside the scene and off the channels."""
    size = terrain.params.size_m
    center_u, center_v = _positions(rng, terrain, n_clusters, 10.0, 8.0)
    per_cluster = rng.poisson(terrain.params.shrubs_per_cluster, n_clusters)
    shrub_u = np.repeat(center_u, per_cluster) + rng.normal(0.0, 9.0, int(per_cluster.sum()))
    shrub_v = np.repeat(center_v, per_cluster) + rng.normal(0.0, 9.0, int(per_cluster.sum()))
    ok = (
        (shrub_u >= 0)
        & (shrub_u < size)
        & (shrub_v >= 0)
        & (shrub_v < size)
        & (np.abs(shrub_v - terrain.channel_v(shrub_u)) > 6.0)
        & (terrain.tributary_distance(shrub_u, shrub_v) > 5.0)
    )
    return shrub_u[ok], shrub_v[ok]


def place_plants(rng: np.random.Generator, terrain: SyntheticTerrain) -> Plants:
    """Scatter trees, small trees near and away from the channel, and shrub patches.

    Args:
        rng: Random generator.
        terrain: The ground the plants stand on.

    Returns:
        All crowns.
    """
    p = terrain.params
    area_ha = p.size_m * p.size_m / 10_000.0
    n_trees = int(rng.poisson(p.trees_per_ha * area_ha))
    tree_u, tree_v = _positions(rng, terrain, n_trees, 15.0, 8.0)
    n_small = int(rng.poisson(p.small_trees_per_ha * area_ha))
    small_u, small_v = _small_tree_positions(rng, terrain, n_small)
    n_clusters = int(rng.poisson(p.shrub_clusters_per_ha * area_ha))
    shrub_u, shrub_v = _shrub_positions(rng, terrain, n_clusters)
    counts = (len(shrub_u), len(small_u), len(tree_u))
    return Plants(
        u=np.concatenate([shrub_u, small_u, tree_u]),
        v=np.concatenate([shrub_v, small_v, tree_v]),
        height=np.concatenate(
            [
                rng.uniform(0.8, 1.8, counts[0]),
                rng.uniform(2.4, 4.6, counts[1]),
                rng.uniform(6.0, 12.0, counts[2]),
            ]
        ),
        radius=np.concatenate(
            [
                rng.uniform(0.9, 2.0, counts[0]),
                rng.uniform(1.5, 2.8, counts[1]),
                rng.uniform(3.0, 5.5, counts[2]),
            ]
        ),
        kind=np.repeat(np.array([SHRUB, SMALL_TREE, TREE], dtype=np.int64), counts),
    )


def canopy_surface(plants: Plants, size_m: float, cell_m: float) -> FloatArray:
    """Crown-surface height at the center of each cell of a square raster.

    Args:
        plants: Crowns.
        size_m: Scene side in meters.
        cell_m: Cell size in meters.

    Returns:
        Canopy height per cell, row 0 at the north.
    """
    n_cells = round(size_m / cell_m)
    surface = np.zeros((n_cells, n_cells))
    for u, v, h, r in zip(plants.u, plants.v, plants.height, plants.radius, strict=True):
        c0, c1 = max(0, int((u - r) / cell_m)), min(n_cells, int((u + r) / cell_m) + 1)
        r0 = max(0, int((size_m - v - r) / cell_m))
        r1 = min(n_cells, int((size_m - v + r) / cell_m) + 1)
        if c0 >= c1 or r0 >= r1:
            continue
        cu = (np.arange(c0, c1) + 0.5) * cell_m
        cv = size_m - (np.arange(r0, r1) + 0.5) * cell_m
        heights = crown_height(cu[None, :] - u, cv[:, None] - v, float(h), float(r))
        surface[r0:r1, c0:c1] = np.maximum(surface[r0:r1, c0:c1], heights)
    return surface


def block_max(fine: FloatArray, factor: int) -> FloatArray:
    """Maximum over each ``factor x factor`` block of a square raster."""
    n = fine.shape[0] // factor
    return np.asarray(fine.reshape(n, factor, n, factor).max(axis=(1, 3)), dtype=np.float64)


def _canopy_points(
    rng: np.random.Generator, plants: Plants, params: SceneParams
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Crown-surface and interior returns; a surface return hidden by a taller crown is dropped."""
    centers = cKDTree(np.column_stack([plants.u, plants.v]))
    r_max = float(plants.radius.max())
    us: list[FloatArray] = []
    vs: list[FloatArray] = []
    hs: list[FloatArray] = []
    for i in range(len(plants)):
        u0, v0 = float(plants.u[i]), float(plants.v[i])
        h, r = float(plants.height[i]), float(plants.radius[i])
        area = math.pi * r * r
        n_top = int(rng.poisson(params.canopy_top_density * area))
        dist = r * np.sqrt(rng.random(n_top))
        angle = rng.uniform(0.0, 2 * math.pi, n_top)
        pu, pv = u0 + dist * np.cos(angle), v0 + dist * np.sin(angle)
        own = crown_height(pu - u0, pv - v0, h, r)
        taller = np.zeros(n_top)
        for j in centers.query_ball_point([u0, v0], r + r_max):
            if j != i:
                other = crown_height(
                    pu - plants.u[j],
                    pv - plants.v[j],
                    float(plants.height[j]),
                    float(plants.radius[j]),
                )
                taller = np.maximum(taller, other)
        visible = own >= taller
        # Returns land slightly below the true crown top, never above it.
        top = own - np.abs(rng.normal(0.0, 0.05, n_top))
        us.append(pu[visible])
        vs.append(pv[visible])
        hs.append(np.clip(top[visible], 0.0, None))

        n_inner = int(rng.poisson(params.canopy_interior_density * area))
        dist = r * np.sqrt(rng.random(n_inner))
        angle = rng.uniform(0.0, 2 * math.pi, n_inner)
        iu, iv = u0 + dist * np.cos(angle), v0 + dist * np.sin(angle)
        inner = crown_height(iu - u0, iv - v0, h, r) * rng.uniform(0.15, 0.95, n_inner)
        us.append(iu)
        vs.append(iv)
        hs.append(inner)
    u, v, hag = np.concatenate(us), np.concatenate(vs), np.concatenate(hs)
    inside = (u >= 0) & (u < params.size_m) & (v >= 0) & (v < params.size_m) & (hag > 0.02)
    return u[inside], v[inside], hag[inside]


def _zone_collection(size_m: float) -> dict[str, Any]:
    s = size_m
    local = {
        "Unit A": [(0, 0), (0.38 * s, 0), (0.30 * s, s), (0, s)],
        "Unit B": [(0.38 * s, 0), (0.72 * s, 0), (0.66 * s, s), (0.30 * s, s)],
        "Unit C": [(0.72 * s, 0), (s, 0), (s, s), (0.66 * s, s)],
    }
    x0, y0 = SYNTHETIC_ORIGIN_XY
    features = []
    for name, ring in local.items():
        closed = [*ring, ring[0]]
        lon, lat = transform_points(
            SYNTHETIC_CRS,
            "EPSG:4326",
            [x0 + u for u, _ in closed],
            [y0 + v for _, v in closed],
        )
        coords = [[round(a, 8), round(b, 8)] for a, b in zip(lon, lat, strict=True)]
        features.append(
            {
                "type": "Feature",
                "properties": {"name": name, "note": "synthetic management unit"},
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _validate(p: SceneParams) -> None:
    cells = p.size_m / p.res_m if p.res_m > 0 else 0.0
    if p.size_m < MIN_SCENE_SIZE_M or p.res_m <= 0 or abs(cells - round(cells)) > 1e-9:
        raise ConfigError(
            f"scene size must be at least {MIN_SCENE_SIZE_M:g} m and a whole number of "
            f"{p.res_m:g} m cells, got {p.size_m:g} m"
        )
    densities = (
        p.ground_density,
        p.canopy_top_density,
        p.canopy_interior_density,
        p.trees_per_ha,
        p.small_trees_per_ha,
        p.shrub_clusters_per_ha,
        p.shrubs_per_cluster,
    )
    if min(densities) < 0 or p.noise_points < 0 or p.truth_supersample < 1:
        raise ConfigError("densities and counts must be non-negative")


def _ground_points(
    rng: np.random.Generator, p: SceneParams, fine_truth: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Ground returns; under a crown only a share of them get through."""
    size = p.size_m
    fine_res = p.res_m / p.truth_supersample
    n_ground = int(rng.poisson(p.ground_density * size * size))
    gu = rng.uniform(0.0, size, n_ground)
    gv = rng.uniform(0.0, size, n_ground)
    last = fine_truth.shape[0] - 1
    fine_row = np.minimum(((size - gv) / fine_res).astype(np.int64), last)
    fine_col = np.minimum((gu / fine_res).astype(np.int64), last)
    covered = fine_truth[fine_row, fine_col] > 0.05
    keep = ~covered | (rng.random(n_ground) < p.ground_keep_under_canopy)
    gu, gv = gu[keep], gv[keep]
    return gu, gv, rng.normal(0.0, p.ground_noise_m, len(gu))


def _noise_points(
    rng: np.random.Generator, p: SceneParams
) -> tuple[FloatArray, FloatArray, FloatArray, UInt8Array]:
    """Returns far above or below the ground, flagged as noise the way providers do."""
    n = p.noise_points
    nu = rng.uniform(0.0, p.size_m, 2 * n)
    nv = rng.uniform(0.0, p.size_m, 2 * n)
    hag = np.concatenate([rng.uniform(25.0, 60.0, n), -rng.uniform(2.0, 6.0, n)])
    return nu, nv, hag, np.repeat(np.array([HIGH_NOISE, LOW_NOISE], dtype=np.uint8), n)


def make_scene(seed: int = 7, params: SceneParams | None = None) -> SyntheticScene:
    """Generate a synthetic scene.

    Args:
        seed: Seed for ``numpy.random.default_rng``; the same seed gives the same scene.
        params: Generation settings (defaults to :class:`SceneParams`).

    Returns:
        Points plus truth rasters, crowns, channel centerline and zones.

    Raises:
        ConfigError: If the scene is too small or a density is negative.
    """
    p = params or SceneParams()
    _validate(p)
    rng = np.random.default_rng(seed)
    size = p.size_m
    terrain = SyntheticTerrain(p, rng)
    plants = place_plants(rng, terrain)
    fine_truth = canopy_surface(plants, size, p.res_m / p.truth_supersample)
    gu, gv, ground_hag = _ground_points(rng, p, fine_truth)
    cu, cv, canopy_hag = _canopy_points(rng, plants, p)
    nu, nv, noise_hag, noise_class = _noise_points(rng, p)

    n_signal = len(gu) + len(cu)
    classification = np.concatenate([np.full(n_signal, UNCLASSIFIED, dtype=np.uint8), noise_class])
    is_ground = np.zeros(n_signal + len(nu), dtype=bool)
    is_ground[: len(gu)] = True
    order = rng.permutation(len(classification))
    u = np.concatenate([gu, cu, nu])[order]
    v = np.concatenate([gv, cv, nv])[order]
    hag = np.concatenate([ground_hag, canopy_hag, noise_hag])[order]

    x0, y0 = SYNTHETIC_ORIGIN_XY
    n_cells = round(size / p.res_m)
    grid = Grid(x0, y0 + size, p.res_m, n_cells, n_cells, SYNTHETIC_CRS, 1.0)
    cloud = PointCloud(
        x0 + u,
        y0 + v,
        terrain.elevation(u, v) + hag,
        classification[order],
        crs=SYNTHETIC_CRS,
        meters_per_unit=1.0,
        extent=grid.bounds,
    )
    centers = (np.arange(n_cells) + 0.5) * p.res_m
    uu, vv = np.meshgrid(centers, size - centers)
    channel_u = np.linspace(0.0, size, 4001)
    return SyntheticScene(
        cloud=cloud,
        grid=grid,
        params=p,
        seed=seed,
        is_ground=is_ground[order],
        height_above_ground=hag,
        ground_true=terrain.elevation(uu, vv),
        chm_true=block_max(fine_truth, p.truth_supersample),
        plants=plants,
        channel_uv=np.column_stack([channel_u, terrain.channel_v(channel_u)]),
        zones=_zone_collection(size),
    )
