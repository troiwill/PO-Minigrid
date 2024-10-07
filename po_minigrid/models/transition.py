from __future__ import annotations
from typing import Any, Callable, Optional

from minigrid.core.actions import Actions
import numpy as np
import numpy.typing as npt

from po_minigrid.models.noise import DiscreteNoiseModel


class MotionTransitionModel:
    """A motion transition model that uses discrete states and stochastic motions."""

    def __init__(self, discrete_noise: npt.ArrayLike | None = None) -> None:
        """Initializes the MotionTransitionModel with a noise matrix.

        Args:
            discrete_noise (npt.ArrayLike): A 2D square matrix of noise probabilities.

        Raises:
            ValueError: If the discrete noise is not a 2D square matrix.
        """
        super().__init__()
        self._discrete_noise: np.ndarray | None = None
        if discrete_noise is not None:
            self._discrete_noise = np.array(discrete_noise, dtype=float)
            if len(self._discrete_noise.shape) != 2:
                raise ValueError("The discrete noise must be 2D.")
            if self._discrete_noise.shape[0] != self._discrete_noise.shape[1]:
                raise ValueError("The discrete noise must be a square matrix.")
            if (noise_sum := np.sum(self._discrete_noise)) != 1.0:
                raise ValueError(
                    f"The discrete noise must sum to 1.0! Sum = {noise_sum}."
                )

    def sample(
        self,
        state: npt.ArrayLike,
        motion: Actions,
        in_goal_predicate: Optional[Callable[[Any], bool]] = None,
        in_failure_predicate: Optional[Callable[[Any], bool]] = None,
        **kwargs,
    ) -> npt.ArrayLike:
        """Samples the next state based on the current state, a motion command, and noise.

        Args:
            state (ArrayLike): The current state of the object.
            motion (EnvMoveLocalizeActions): The motion command (left, right, forward).
            **kwargs: Arbitrary keyword arguments.

        Returns:
            npt.ArrayLike: The new state after the motion and noise have been applied.
        """
        # Sanity checks.
        assert motion in (
            Actions.forward,
            Actions.left,
            Actions.right,
        ), f"Unhandled motion? {motion}"

        # Check if the state is in the goal or failure state.
        pos, theta = state[:2], state[2]
        if in_goal_predicate is not None and in_goal_predicate(pos):
            return state

        if in_failure_predicate is not None and in_failure_predicate(pos):
            return state

        # Add noise to the next state.
        if motion == Actions.forward:
            if self._discrete_noise is not None:
                matrix_rot_angle = theta if theta % 2 == 0 else (theta + 2) % 4
                pos = DiscreteNoiseModel.sample(
                    pos, np.rot90(self._discrete_noise, k=matrix_rot_angle)
                )

            else:
                delta_x, delta_y = 0, 0
                if theta == 0:
                    delta_x = 1
                elif theta == 1:
                    delta_y = 1
                elif theta == 2:
                    delta_x = -1
                elif theta == 3:
                    delta_y = -1
                pos = (state[0] + delta_x, state[1] + delta_y)

        elif motion == Actions.left:
            theta -= 1
            if theta < 0:
                theta += 4

        # Rotate right
        elif motion == Actions.right:
            theta = (theta + 1) % 4

        else:
            raise Exception(
                f"Unhandled motion command? {motion}. It missed the sanity check!"
            )

        return (pos[0], pos[1], theta)
