import pytest

import numpy as np

from po_minigrid.core.particles import Particles
from po_minigrid.models.noise_base import DiscreteNoiseModel
from po_minigrid.models.transition.noise import NoisyForward


@pytest.mark.parametrize(
    "theta,potential_offsets",
    [
        (
            0,
            [
                np.array([0, 1]),
                np.array([0, 0]),
            ],
        ),
        (
            1,
            [
                np.array([1, 0]),
                np.array([0, 0]),
            ],
        ),
        (
            2,
            [
                np.array([0, -1]),
                np.array([0, 0]),
            ],
        ),
        (
            3,
            [
                np.array([-1, 0]),
                np.array([0, 0]),
            ],
        ),
    ],
)
def test_noisy_forward(theta: int, potential_offsets: np.ndarray):
    states = np.repeat(np.array([[0, 0, theta]]), 1_000, axis=0)
    particles = Particles(states)
    noise = np.zeros((3, 3))
    noise[1, 0] = noise[1, 1] = 0.5

    noise_model = NoisyForward(DiscreteNoiseModel(noise))
    noisy_offsets = noise_model(particles).position
    assert np.alltrue(
        [
            np.any([np.equal(noff, poff) for poff in potential_offsets])
            for noff in noisy_offsets
        ]
    )
