"""Shared NumPy array type aliases used across the package."""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Int32Array = npt.NDArray[np.int32]
UInt8Array = npt.NDArray[np.uint8]
BoolArray = npt.NDArray[np.bool_]
