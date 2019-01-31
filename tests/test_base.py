from resample.base import Resampler
from KDEpy.FFTKDE import FFTKDE
import numpy as np
import pytest


@pytest.fixture
def uniform():
    data = np.random.uniform(size=1000)
    return data


def test_resampler_init(uniform):
    data = uniform
    resampler = Resampler(data=data)
    assert isinstance(resampler, Resampler)
    assert isinstance(resampler.kde, FFTKDE)
    assert len(data) == resampler.num_samples


def test_resampler_find_indices(uniform):
    data = uniform
    resampler = Resampler(data=data)
    for num_samples in [10, 100, 1000, 10_000]:
        indices = resampler.find_indices(num_samples=num_samples)
        assert len(indices) == num_samples
        assert np.all([
            x in range(len(data))
            for x in indices
        ])
