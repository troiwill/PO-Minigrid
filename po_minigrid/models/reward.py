from __future__ import annotations
from typing import Any

from minigrid.core.grid import Grid
import numpy as np

from po_minigrid.models.utils import get_grid_cell


class RewardModel:
    """A class representing a reward model.

    This model outputs rewards based on the state of the agent.
    """

    def sample(
        self,
        states: np.ndarray,
        action: Any,
        grid: Grid,
        **kwargs,
    ) -> np.ndarray:
        # Determine if the state(s) are at the goal.
        curr_cells = get_grid_cell(grid=grid, position=states[:, :2])
        rewards = np.array(
            [
                10.0 if cell is not None and cell.type == "goal" else 0.0
                for cell in curr_cells
            ]
        )

        return rewards
