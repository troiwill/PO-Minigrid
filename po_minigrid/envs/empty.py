from __future__ import annotations

import time

from minigrid.core.actions import Actions
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
import numpy as np

from po_minigrid.models.transition import MotionTransitionModel
from po_minigrid.po_minigrid_env import POMiniGridEnv
from po_minigrid.core.po_grid import POGrid


class POEmptyEnv(POMiniGridEnv):
    """
    ## Description

    This environment is an empty room, and the goal of the agent is to reach the
    green goal square, which provides a sparse reward. A small penalty is
    subtracted for the number of steps to reach the goal. This environment is
    useful, with small rooms, to validate that your RL algorithm works
    correctly, and with large rooms to experiment with sparse rewards and
    exploration. The random variants of the environment have the agent starting
    at a random position for each episode, while the regular variants have the
    agent always starting in the corner opposite to the goal.

    ## Mission Space

    "get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-Empty-5x5-v0`
    - `MiniGrid-Empty-Random-5x5-v0`
    - `MiniGrid-Empty-6x6-v0`
    - `MiniGrid-Empty-Random-6x6-v0`
    - `MiniGrid-Empty-8x8-v0`
    - `MiniGrid-Empty-16x16-v0`

    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = POGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"


def main():
    start_state: tuple[int, int, int] = (2, 2, 0)
    grid_size = 9
    env = POEmptyEnv(
        render_mode="human",
        size=grid_size,
        agent_start_pos=start_state[:2],
        agent_start_dir=start_state[2],
    )
    env._gen_grid(grid_size, grid_size)

    particles = np.full((100, 3), start_state, dtype=int).tolist()
    noise = np.zeros((3, 3))
    noise[0, 2] = noise[2, 2] = 0.05
    noise[1, 2] = 0.9
    motion_model = MotionTransitionModel(noise)

    for _ in range(4):
        # Move four steps.
        for _ in range(4):
            env.step(Actions.forward)
            particles = list(motion_model.sample(p, Actions.forward) for p in particles)
            env.render(particles=particles)

        env.step(Actions.right)
        particles = list(motion_model.sample(p, Actions.right) for p in particles)
        env.render(particles=particles)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
