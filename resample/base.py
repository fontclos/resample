from sklearn.neighbors import KernelDensity
import numpy as np
from KDEpy.FFTKDE import FFTKDE
from typing import Sequence, Union

from .utils import map_to_rounded_grid


class Resampler:
    def __init__(
        self,
        data: Sequence,
        drop_tails: float = 0.005,
        decimals: int = 4
    ):
        assert drop_tails <= 0.1
        if np.array(data).ndim != 1:
            raise RuntimeError("Data shape not ok")

        self.data = np.array(data)
        self.kde = FFTKDE(kernel="box", bw="scott").fit(data)
        self.num_samples = len(data)
        self.drop_tails = drop_tails
        self._decimals = decimals
        # set xmin, xmax to avoid noisy tails
        self._set_xmin_xmax()
        # estimate the density at the datapoints using a grid
        self._estimate_density()

    def find_indices(self, num_samples: int):
        return np.random.choice(
            len(self.data),
            size=num_samples,
            p=self._data_density
        )

    # PRIVATE METHODS
    def _estimate_density(self):
        grid, indices = map_to_rounded_grid(
            data=self.data,
            decimals=self._decimals
        )
        grid_density = self.kde.evaluate(grid)
        data_density = 1 / grid_density[indices]

        # set prob of tails to zero
        for i, x in enumerate(self.data):
            if x < self.xmin or x > self.xmax:
                data_density[i] = 0
        self._data_density = data_density / sum(data_density)

    def _set_xmin_xmax(self):
        m, M = np.percentile(
            self.data,
            [100 * self.drop_tails, 100 - 100 * self.drop_tails]
        )
        self.xmin = m
        self.xmax = M
