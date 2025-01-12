from __future__ import annotations

import numpy as np
from minigrid.core.grid import Grid

from po_minigrid.core.particles import Particles
from po_minigrid.models.noise_base import NoiseModel
from po_minigrid.models.utils import can_enter_cell, get_grid_cell


class NoisyForward:
    """Represents a noisy forward action in a grid-based environment.

    This class applies a noise model to the forward movement of agents,
    potentially altering their positions based on the noise and environment constraints.
    """

    def __init__(self, noise_model: NoiseModel) -> None:
        """Initializes the NoisyForward instance with a noise model.

        Args:
            noise_model: An instance of NoiseModel to generate position offsets.

        Raises:
            AssertionError: If the provided noise_model is not an instance of NoiseModel.
        """
        assert isinstance(noise_model, NoiseModel)
        self.noise_model: NoiseModel = noise_model

    def __call__(
        self, particles: Particles, grid: Grid | None = None, **kwargs
    ) -> Particles:
        """Applies noisy forward movement to the given particles.

        This method computes noisy positions for the agents and updates their positions
        if the new positions are valid within the grid.

        Args:
            particles: A Particles instance representing the current states of the agents.
            grid: An optional Grid object representing the environment. If provided,
                  it's used to check if the noisy positions are valid.
            **kwargs: Additional keyword arguments (unused in this method).

        Returns:
            A Particles instance containing the updated states
            after applying noisy forward movement.

        Raises:
            AssertionError: If the pose array of the input Particles doesn't have the expected shape.
        """
        # Sanity checks.
        pose = particles.pose
        assert len(pose.shape) == 2 and pose.shape[1] == 3

        # Compute a noisy position.
        sampled_offset: np.ndarray = self.noise_model.sample(thetas=pose[:, 2])
        noisy_pos = pose[:, :2] + sampled_offset.reshape(-1, 2)

        # Determine if we can use the noisy position or the original position.
        if grid is not None:
            noisy_cells = get_grid_cell(grid, noisy_pos)
            can_enter_cell_bools = can_enter_cell(noisy_cells)
        else:
            can_enter_cell_bools = np.repeat([True], len(pose))
        particles.position = np.where(
            np.expand_dims(can_enter_cell_bools, axis=1), noisy_pos, pose[:, :2]
        )
        return particles
