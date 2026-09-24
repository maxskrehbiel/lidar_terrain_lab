"""Exception types the package raises on purpose, so callers can tell bad input from bugs."""


class LidarTerrainLabError(Exception):
    """Base class for every error this package raises deliberately."""


class ConfigError(LidarTerrainLabError, ValueError):
    """A configuration file or setting is malformed, unknown or out of range."""


class InputDataError(LidarTerrainLabError, ValueError):
    """Input data (a point cloud or a zone file) is missing, unreadable or unusable."""


class FetchError(LidarTerrainLabError, RuntimeError):
    """The USGS TNM API or a tile download failed or returned something unexpected."""


class LidarDependencyError(LidarTerrainLabError, ImportError):
    """LAS/LAZ support was requested but the optional laspy dependency is not installed."""
