from resample.base import Resampler
from sklearn.neighbors import KernelDensity
import numpy as np
import pytest


@pytest.fixture
def uniform():
    data = np.random.uniform(size=1000)
    return data


def test_resampler_init(uniform):
    for data in [uniform, list(uniform), np.array([uniform]).T]:
        resampler = Resampler(data=data)
        assert isinstance(resampler, Resampler)
        assert isinstance(resampler.kde, KernelDensity)
        assert len(data) == resampler.num_samples
