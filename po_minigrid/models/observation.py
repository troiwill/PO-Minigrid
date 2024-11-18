from __future__ import annotations

from minigrid.core.grid import Grid
import numpy as np

from po_minigrid.models.noise_base import DiscreteNoiseModel
from po_minigrid.models.utils import can_enter_cell, get_grid_cell


class ObservationModel:
    """Represents an observation model for a grid-based environment.

    This class handles the generation of potentially noisy observations
    based on the true states of agents in the environment.
    """

    def __init__(self, noise_model: DiscreteNoiseModel | None = None) -> None:
        """Initializes the ObservationModel with an optional noise model.

        Args:
            noise_model: An optional DiscreteNoiseModel instance to generate noisy observations.
                         If None, the model will return states without noise.
        """
        self.noise_model = noise_model

    def sample(
        self, states: np.ndarray, grid: Grid | None = None, **kwargs
    ) -> np.ndarray:
        """Samples observations based on the true states.

        This method generates observations by either returning the unaltered states
        or applying noise to the positions based on the noise model.

        Args:
            states: A numpy array of shape (n, 3) representing the states.
                    Each row contains [x, y, theta] for an agent.
            grid: An optional Grid object representing the environment. If provided,
                  it's used to check if the noisy positions are valid.
            **kwargs: Additional keyword arguments (unused in this method).

        Returns:
            A numpy array of the same shape as 'states', containing the sampled observations.
            If no noise model is set, this will be identical to the input states.

        Notes:
            - If a noise model is set, it generates noisy positions and checks their validity
              against the provided grid.
            - Invalid noisy positions are replaced with the original positions.
        """
        # Return the state(s) if there is no noise model.
        if self.noise_model is None:
            return states

        # Generate a noisy observation if there is a noise model.
        noisy_offsets = self.noise_model.sample(states[:, 2])
        noisy_pos = states[:, :2] + noisy_offsets.reshape(-1, 2)

        # Determine if we can use the noisy position or the original position.
        noisy_cells = get_grid_cell(grid, noisy_pos)
        can_enter_cell_bools = can_enter_cell(noisy_cells)
        states[:, :2] = np.where(
            np.expand_dims(can_enter_cell_bools, axis=1), noisy_pos, states[:, :2]
        )

        return states
