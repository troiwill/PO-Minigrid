from __future__ import annotations

from minigrid.core.grid import Grid
import numpy as np

from po_minigrid.models.utils import (
    headings_are_valid,
    get_grid_cell,
    compute_fwd_pos,
    can_enter_cell,
)


class ActionLeft:
    """Represents an action that turns the agent left.

    This class implements the left turn action for agents in a grid-based environment.
    """

    def __call__(self, states: np.ndarray, **kwargs) -> np.ndarray:
        """Applies a left turn to the given states.

        Args:
            states: A numpy array representing the current states of the agents.
                    The third column (index 2) contains the heading information.
            **kwargs: Additional keyword arguments (unused in this method).

        Returns:
            A numpy array with updated states after applying the left turn.

        Raises:
            AssertionError: If the headings in the states are not valid.
        """
        thetas = states[:, 2]
        assert headings_are_valid(thetas)

        thetas -= 1
        thetas[thetas < 0] += 4
        states[:, 2] = thetas
        return states


class ActionRight:
    """Represents an action that turns the agent right.

    This class implements the right turn action for agents in a grid-based environment.
    """

    def __call__(self, states: np.ndarray, **kwargs) -> np.ndarray:
        """Applies a right turn to the given states.

        Args:
            states: A numpy array representing the current states of the agents.
                    The third column (index 2) contains the heading information.
            **kwargs: Additional keyword arguments (unused in this method).

        Returns:
            A numpy array with updated states after applying the right turn.

        Raises:
            AssertionError: If the headings in the states are not valid.
        """
        thetas = states[:, 2]
        assert headings_are_valid(thetas)

        thetas = (thetas + 1) % 4
        states[:, 2] = thetas
        return states


class ActionForward:
    """Represents an action that moves the agent forward.

    This class implements the forward movement action for agents in a grid-based environment.
    """

    def __init__(
        self,
        ignore_fwd_cell_valid_check: bool = False,
    ) -> None:
        """Initializes the ActionForward instance.

        Args:
            ignore_fwd_cell_valid_check: If True, allows movement through walls or
                out of the environment. Defaults to False.
        """
        self._ignore_fwd_cell_valid_check = ignore_fwd_cell_valid_check

    def __call__(self, states: np.ndarray, grid: Grid, **kwargs) -> np.ndarray:
        """Applies a forward movement to the given states.

        Args:
            states: A numpy array representing the current states of the agents.
            grid: The grid object representing the environment.
            **kwargs: Additional keyword arguments (unused in this method).

        Returns:
            A numpy array with updated states after applying the forward movement.
        """
        # Compute the forward position of each state.
        fwd_pos = compute_fwd_pos(states)

        # If we ignore this check, the forward state can "go through" walls or out the environment.
        if self._ignore_fwd_cell_valid_check:
            states[:, :2] = fwd_pos
        else:
            fwd_cells = get_grid_cell(grid, fwd_pos)
            can_enter_cell_bools = can_enter_cell(fwd_cells)
            states[:, :2] = np.where(
                np.expand_dims(can_enter_cell_bools, axis=1), fwd_pos, states[:, :2]
            )
        return states
