from sklearn.neighbors import KernelDensity
import numpy as np

from typing import Sequence, Union


class Resampler:
    def __init__(
        self,
        data: Sequence,
        density_estimator: KernelDensity = KernelDensity(
            bandwidth = 0.1,
        ),
        drop_tails: float = 0.005
    ):
        assert drop_tails <= 0.1
        data = np.array(data)
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        elif len(data.shape) == 2 and (data.shape[0] == 1 or data.shape[1] != 1):
            raise RuntimeError("Data shape not ok")
        elif len(data.shape) > 2:
            raise RuntimeError("Data shape not ok")

        self.kde = density_estimator
        self.kde.fit(data)
        self.data = data
        self.num_samples = len(data)
        self.drop_tails = drop_tails
        # internal computations
        self._drop_tails()
        self._compute_probs()

    def resample(self, num_samples: int):
        return np.random.choice(
            self.filtered_data.T[0],
            size=num_samples,
            p=self._probs
        )

    # PRIVATE METHODS
    def _compute_probs(self):
        self._logprobs = self.kde.score_samples(self.filtered_data)
        probs = 1./np.exp(self._logprobs)
        self._probs = probs / sum(probs)

    def _drop_tails(self):
        m, M = np.percentile(
            self.data.T[0],
            [100 * self.drop_tails, 100 - 100 * self.drop_tails]
        )
        self.xmin = m
        self.xmax = M
        filtered_data = [
            x
            for x in self.data.T[0]
            if x >= m and x <= M
        ]
        filtered_data = np.array([filtered_data]).T
        self.filtered_data = filtered_data
