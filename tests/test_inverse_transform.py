import numpy as np

from mc_lab.inverse_transform import (
    create_sampler,
)


def test_create_sampler():
    def exp_inverse_cdf(u, rate=2.0):
        return -np.log(1 - u) / rate

    # Create analytical sampler
    exp_sampler = create_sampler(
        inverse_cdf=lambda u: exp_inverse_cdf(u, rate=2.0),
        method="analytical",
        random_state=42,
    )

    samples = exp_sampler.sample(100000)

    assert np.isclose(np.mean(samples), 0.5, atol=0.1)
