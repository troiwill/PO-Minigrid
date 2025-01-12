from __future__ import annotations

from typing import Any

import numpy as np
from minigrid.core.grid import Grid

from po_minigrid.core.particles import Particles
from po_minigrid.models.utils import get_grid_cell


class RewardModel:
    """A class representing a reward model.

    This model outputs rewards based on the state of the agent.
    """

    def sample(
        self,
        particles: Particles,
        action: Any,
        grid: Grid,
        **kwargs,
    ) -> np.ndarray:
        # Determine if the state(s) are at the goal.
        curr_cells = get_grid_cell(grid=grid, position=particles.position)
        rewards = np.array(
            [
                10.0 if cell is not None and cell.type == "goal" else 0.0
                for cell in curr_cells
            ]
        )

        return rewards
