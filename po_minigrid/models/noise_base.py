from __future__ import annotations

from itertools import product

import numpy as np


class NoiseModel:
    """Base class for noise models.

    This abstract class defines the interface for noise models used in the environment.
    """

    def sample(self, **kwargs) -> np.ndarray:
        """Samples noise from the model.

        Args:
            **kwargs: Additional keyword arguments specific to the noise model.

        Returns:
            A numpy array representing the sampled noise.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError


class DiscreteNoiseModel(NoiseModel):
    """A discrete noise model for generating position offsets.

    This class implements a discrete noise model that samples position offsets
    based on a given noise distribution and agent orientations.
    """

    def __init__(self, noise: np.ndarray) -> None:
        """Initializes the DiscreteNoiseModel with a given noise distribution.

        Args:
            noise: A 2D numpy array representing the noise distribution.

        Raises:
            AssertionError: If the noise array doesn't meet the required conditions.
        """
        super().__init__()
        self._discrete_noise: np.ndarray
        self.discrete_noise = noise
        self._vectorized_sample_position_offset = np.vectorize(
            DiscreteNoiseModel._sample_position_offset, signature="(),(n,n)->(m)"
        )

    @property
    def discrete_noise(self) -> np.ndarray:
        """Gets a copy of the discrete noise distribution.

        Returns:
            A copy of the discrete noise distribution array.
        """
        return self._discrete_noise.copy()

    @discrete_noise.setter
    def discrete_noise(self, noise: np.ndarray) -> None:
        """Sets the discrete noise distribution.

        Args:
            noise: A 2D numpy array representing the new noise distribution.

        Raises:
            AssertionError: If the noise array doesn't meet the required conditions.
        """
        assert len(noise.shape) == 2
        assert noise.shape[0] == noise.shape[1] and noise.shape[0] % 2 == 1
        assert np.sum(noise) == 1.0
        self._discrete_noise = noise.copy()

    @staticmethod
    def _sample_position_offset(
        theta: int,
        noise: np.ndarray,
    ) -> np.ndarray:
        """Samples a position offset based on the agent's orientation and noise distribution.

        Args:
            theta: An integer representing the agent's orientation (0, 1, 2, or 3).
            noise: A 2D numpy array representing the noise distribution.

        Returns:
            A numpy array representing the sampled position offset.
        """
        matrix_rot_angle_dict = {0: -1, 1: 2, 2: 1, 3: 0}
        if (rot_angle := matrix_rot_angle_dict[theta]) != 0:
            noise = np.rot90(noise, k=rot_angle)

        half_dim = noise.shape[0] // 2
        noise_matrix_offsets = np.fliplr(
            np.array(
                list(
                    product(
                        np.arange(-half_dim, half_dim + 1),
                        np.arange(-half_dim, half_dim + 1),
                    )
                )
            ).reshape(-1, 2)
        )

        sampled_index = np.random.choice(
            range(len(noise_matrix_offsets)),
            size=1,
            replace=True,
            p=noise.flatten(),
        )

        sampled_offset = noise_matrix_offsets[sampled_index.flatten().tolist()]
        return sampled_offset

    def sample(self, thetas: np.ndarray, **kwargs) -> np.ndarray:
        """Samples position offsets for multiple agents based on their orientations.

        Args:
            thetas: A numpy array of integers representing the orientations of multiple agents.
            **kwargs: Additional keyword arguments (unused in this method).

        Returns:
            A numpy array of sampled position offsets for each agent.
        """
        return self._vectorized_sample_position_offset(thetas, self._discrete_noise)
