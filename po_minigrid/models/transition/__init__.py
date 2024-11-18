from __future__ import annotations
from typing import Callable, Any

from minigrid.core.actions import Actions
from minigrid.core.grid import Grid
import numpy as np

from po_minigrid.models.transition.deterministic import (
    ActionForward,
    ActionLeft,
    ActionRight,
)

ActionFnType = Callable[..., np.ndarray]
NoiseFnType = ActionFnType


class TransitionModel:
    """Represents a transition model for a grid-based environment.

    This class handles the application of actions and noise to agent states.
    """

    def __init__(
        self,
        action_fn_dict: dict[Any, ActionFnType] = dict(),
        noise_fn_dict: dict[Any, NoiseFnType] = dict(),
    ) -> None:
        """Initializes the TransitionModel with action and noise functions.

        Args:
            action_fn_dict: A dictionary mapping action types to action functions.
                            Defaults to an empty dictionary.
            noise_fn_dict: A dictionary mapping action types to noise functions.
                           Defaults to an empty dictionary.
        """
        self.action_dict = action_fn_dict.copy()
        if len(self.action_dict) == 0:
            self.action_dict = {
                Actions.right: ActionRight(),
                Actions.left: ActionLeft(),
                Actions.forward: ActionForward(),
            }

        self.noise_dict = noise_fn_dict.copy()
        if len(self.noise_dict) == 0:
            pass

    def sample(
        self,
        states: np.ndarray,
        action: Any,
        grid: Grid,
        soft_fail: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Samples the next state given the current state and action.

        This method applies the specified action to the given states and then
        applies noise if a noise function is defined for the action.

        Args:
            states: A numpy array representing the current state(s) of the agent(s).
            action: The action to be applied to the states.
            grid: The grid object representing the environment.
            soft_fail: If True, prints a warning for unimplemented actions instead of raising an error.
                       Defaults to True.
            **kwargs: Additional keyword arguments to be passed to the action and noise functions.

        Returns:
            A numpy array representing the next state(s) after applying the action and noise.

        Raises:
            NotImplementedError: If the action is not implemented and soft_fail is False.
        """
        # Accounts for the case where we pass the true agent state, which would be a 1D array.
        state_was_flat = False
        if states.shape == (3,):
            state_was_flat = True
            states = states.reshape(1, 3)

        # Apply an action in a deterministic manner.
        if action in self.action_dict:
            states = self.action_dict[action](
                states=states, action=action, grid=grid, **kwargs
            )
        elif soft_fail:
            print(f"WARNING: Action {action} was not implemented.")
            return states
        else:
            raise NotImplementedError(f"Action {action} was not implemented.")

        # Apply noise to the deterministic action.
        if action in self.noise_dict:
            states = self.noise_dict[action](
                states=states, action=action, grid=grid, **kwargs
            )

        if state_was_flat:
            states = states.flatten()
        return states
