from __future__ import annotations

import numpy as np
import numpy.typing as npt

from po_minigrid.models.noise import DiscreteNoiseModel


class ObservationModel:
    """A class representing an observation model with discrete noise.

    This model can apply discrete noise to observations of a state.
    """

    def __init__(self, discrete_noise: npt.ArrayLike | None = None) -> None:
        """Initialize the ObservationModel.

        Args:
            discrete_noise: A 2D array-like object representing the discrete noise.
                Must be a square matrix that sums to 1.0.

        Raises:
            ValueError: If the discrete_noise is not 2D, not square, or doesn't sum to 1.0.
        """
        super().__init__()
        self._discrete_noise: np.ndarray | None = None
        if discrete_noise is not None:
            self._discrete_noise = np.array(discrete_noise, dtype=float)
            if len(self._discrete_noise.shape) != 2:
                raise ValueError("The discrete noise must be 2D.")
            if self._discrete_noise.shape[0] != self._discrete_noise.shape[1]:
                raise ValueError("The discrete noise must be a square matrix.")
            if np.sum(discrete_noise) != 1.0:
                raise ValueError("The discrete noise must sum to 1.0!")

    def sample(self, state: npt.ArrayLike, **kwargs) -> npt.ArrayLike:
        """Sample an observation from the given state.

        If discrete noise is defined, it applies the noise to the state's position.

        Args:
            state: An array-like object representing the current state (x, y, theta).
            **kwargs: Additional keyword arguments (unused in this implementation).

        Returns:
            An array-like object representing the observed state.
        """
        if self._discrete_noise is None:
            return state

        x, y, theta = state
        matrix_rot_angle = theta if theta % 2 == 0 else (theta + 2) % 4
        observd_pos = DiscreteNoiseModel.sample(
            (x, y), np.rot90(self._discrete_noise, k=matrix_rot_angle)
        )
        return (observd_pos[0], observd_pos[1], theta)
