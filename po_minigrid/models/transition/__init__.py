from __future__ import annotations

from typing import Any, Callable

from minigrid.core.actions import Actions
from minigrid.core.grid import Grid

from po_minigrid.core.particles import Particles
from po_minigrid.models.transition.deterministic import (
    ActionForward,
    ActionLeft,
    ActionRight,
)

ActionFnType = Callable[..., Particles]
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
        particles: Particles,
        action: Any,
        grid: Grid,
        soft_fail: bool = True,
        **kwargs,
    ) -> Particles:
        """Samples the next state given the current state and action.

        This method applies the specified action to the given states and then
        applies noise if a noise function is defined for the action.

        Args:
            particles: A Particles object representing the current state(s) of the agent(s).
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
        # Apply an action in a deterministic manner.
        if action in self.action_dict:
            particles = self.action_dict[action](
                particles=particles, action=action, grid=grid, **kwargs
            )
        elif soft_fail:
            print(f"WARNING: Action {action} was not implemented.")
            return particles
        else:
            raise NotImplementedError(f"Action {action} was not implemented.")

        # Apply noise to the deterministic action.
        if action in self.noise_dict:
            particles = self.noise_dict[action](
                particles=particles, action=action, grid=grid, **kwargs
            )

        return particles
