from __future__ import annotations

import numpy as np
from minigrid.core.grid import Grid

from po_minigrid.core.particles import Particles
from po_minigrid.models.utils import (  # can_pickup_cell,
    can_enter_cell,
    compute_fwd_pos,
    get_grid_cell,
    headings_are_valid,
)


class ActionLeft:
    """Represents an action that turns the agent left.

    This class implements the left turn action for agents in a grid-based environment.
    """

    def __call__(self, particles: Particles, **kwargs) -> Particles:
        """Applies a left turn to the given states.

        Args:
            particles: A Particles instance representing the current states of the agents.
            **kwargs: Additional keyword arguments (unused in this method).

        Returns:
            A Particles object with updated states after applying the left turn.

        Raises:
            AssertionError: If the headings in the states are not valid.
        """
        thetas = particles.direction
        thetas -= 1
        thetas[thetas < 0] += 4
        assert headings_are_valid(thetas)

        particles.pose[:, 2] = thetas
        return particles


class ActionRight:
    """Represents an action that turns the agent right.

    This class implements the right turn action for agents in a grid-based environment.
    """

    def __call__(self, particles: Particles, **kwargs) -> Particles:
        """Applies a right turn to the given states.

        Args:
            particles: A Particles instance representing the current states of the agents.
            **kwargs: Additional keyword arguments (unused in this method).

        Returns:
            A Particles object with updated states after applying the right turn.

        Raises:
            AssertionError: If the headings in the states are not valid.
        """
        thetas = particles.direction

        thetas = (thetas + 1) % 4
        assert headings_are_valid(thetas)

        particles.pose[:, 2] = thetas
        return particles


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

    def __call__(self, particles: Particles, grid: Grid, **kwargs) -> Particles:
        """Applies a forward movement to the given particles.

        Args:
            particles: A Particles instance representing the current states of the agents.
            grid: The grid object representing the environment.
            **kwargs: Additional keyword arguments (unused in this method).

        Returns:
            A Particles object with updated states after applying the forward movement.
        """
        # Compute the forward position of each state.
        states = particles.pose
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

        particles.pose = states
        return particles


# class ActionPickup:
#     """Represents an action that picks up an item that is in front of the agent.

#     This class implements the pickup action for agents in a grid-based environment.
#     """

#     def __call__(
#         self, particles: Particles, grid: Grid, manipulate_grid: bool = False, **kwargs
#     ) -> Particles:
#         """Applies a pickup action to the given particles.

#         Args:
#             particles: A Particles instance representing the current states of the agents.
#             grid: The grid object representing the environment.
#             manipulate_grid: If True, the grid is manipulated directly. But this can only be
#                 used when the number of particles is 1.
#             **kwargs: Additional keyword arguments (unused in this method).

#         Returns:
#             A Particles object with updated states after applying the pickup action.
#         """
#         # Sanity checks.
#         if manipulate_grid:
#             assert len(particles) == 1 and particles.is_belief == False

#         # Compute the forward cell for each particle.
#         fwd_pos = compute_fwd_pos(particles.pose)
#         fwd_cell = get_grid_cell(grid, fwd_pos)

#         # Determine which particles can pickup an item.
#         can_pickup_indices = can_pickup_cell(fwd_cell)
#         if len(can_pickup_indices) > 0:
#             particles.carrying.set(
#                 indices=can_pickup_indices, values=fwd_cell[can_pickup_indices]
#             )

#             # Directly manipulate the grid if this is a single particle.
#             if manipulate_grid:
#                 particles.carrying[0].curr_pos = np.array([-1, -1])
#                 fwd_pos = fwd_pos.flatten()
#                 grid.set(fwd_pos[0], fwd_pos[1], None)

#         return particles
