import numpy as np
from typing import List


def map_to_rounded_grid(data: List[float], decimals: int):
    """Round the elements of an array"""
    tol = 1 / 10 ** decimals
    x_rounded = np.around(data, decimals=decimals)
    m, M = min(x_rounded) - tol, max(x_rounded) + tol
    grid = np.arange(m, M + tol, tol)
    indices = np.around((x_rounded - m) / tol).astype(int)
    return grid, indices
