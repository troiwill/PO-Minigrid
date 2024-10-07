from __future__ import annotations
import numpy as np
import numpy.typing as npt
from itertools import product


class NoiseModel:
    """Base class for all noise models.

    Methods:
        sample: Abstract static method to be implemented by subclasses.
    """

    @staticmethod
    def sample(
        v: npt.ArrayLike, sample_probs: npt.ArrayLike, **kwargs
    ) -> tuple[int, ...]:
        """Abstract static method to sample a noise vector based on the given probabilities.

        Args:
            v (ArrayLike): The current state vector from which noise should be applied.
            sample_probs (ArrayLike): A matrix of probabilities associated with each noise vector.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple[int, ...]: A tuple representing the new state vector after noise has been applied.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError


class DiscreteNoiseModel(NoiseModel):
    """Discrete implementation of a NoiseModel that uses a probability matrix to sample noise offsets.

    Inherits from NoiseModel.
    """

    @staticmethod
    def sample(
        v: npt.ArrayLike, sample_probs: npt.ArrayLike, **kwargs
    ) -> tuple[int, ...]:
        """Samples a noise vector using a discrete probability matrix and applies it to a vector.

        This method performs checks to ensure that the probability matrix is square and each dimension is odd,
        then samples from this matrix to determine the noise offset.

        Args:
            v (ArrayLike): The state vector to which noise will be applied.
            sample_probs (ArrayLike): A 2D square matrix of probabilities, each dimension must be odd.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple[int, ...]: A tuple representing the new state vector with noise applied.

        Raises:
            ValueError: If `sample_probs` is not 2D, not square, or dimensions are not odd.
        """
        # Sanity checks.
        sample_probs = np.array(sample_probs)
        if len(sample_probs.shape) != 2:
            raise ValueError("Cannot handle sample probs that are not 2D.")
        if any(d % 2 != 1 for d in sample_probs.shape):
            raise ValueError(
                f"Unhandled sample probability array. Got shape = {sample_probs.shape}."
            )
        if sample_probs.shape[0] != sample_probs.shape[1]:
            raise ValueError("Sample probs must be a square matrix.")

        half_dim = sample_probs.shape[0] // 2
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
            p=sample_probs.flatten(),
        ).item()
        sampled_offset = noise_matrix_offsets[sampled_index]
        return (v[0] + sampled_offset[0], v[1] + sampled_offset[1])
