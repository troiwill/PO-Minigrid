from __future__ import annotations

import time

from minigrid.core.actions import Actions
from minigrid.core.mission import MissionSpace
import numpy as np

from po_minigrid.models.noise_base import DiscreteNoiseModel
from po_minigrid.models.transition import TransitionModel
from po_minigrid.models.transition.noise import NoisyForward
from po_minigrid.po_minigrid_env import POMiniGridEnv
from po_minigrid.core.particles import Particles
from po_minigrid.core.po_grid import POGrid


class POEmptyEnv(POMiniGridEnv):
    """
    ## Description

    This environment is an empty room, where we showcase an agent (red) moving in a square
    pattern. As the agent moves, particles (light blue) begin to disperse with increasing
    uncertainty as the agent moves.

    ## Mission Space

    "run in a loop"

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

    ## Rewards

    None

    ## Termination

    The agent performs a full loop in the form of a square.
    """

    def __init__(
        self,
        size=8,
        agent_start_pose=(1, 1, 0),
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pose = agent_start_pose
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
        return "run in a loop"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = POGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        if self.agent_start_pose is not None:
            self.agent_state.pose = np.array([self.agent_start_pose]).reshape(1, 3)
        else:
            self.place_agent()

        self.mission = POEmptyEnv._gen_mission()


def main():
    # Generate the environment.
    start_state: tuple[int, int, int] = (3, 3, 0)
    grid_size = 11
    env = POEmptyEnv(
        render_mode="human",
        size=grid_size,
        agent_start_pose=start_state,
    )
    env.reset()

    # Create the particle distribution.
    particles = Particles(
        pose=np.full((10_000, 3), start_state, dtype=int), is_belief=True
    )

    # Create the transition model.
    transition_model = TransitionModel()
    noise = np.zeros((3, 3))
    noise[1, 0] = noise[1, 2] = 0.1
    noise[1, 1] = 1.0 - np.sum(noise)
    transition_model.noise_dict[Actions.forward] = NoisyForward(
        DiscreteNoiseModel(noise)
    )

    # Run a sequence of move forward and turn right actions to move in a square.
    for _ in range(4):
        for _ in range(4):
            env.step(Actions.forward)
            particles = transition_model.sample(
                particles=particles, action=Actions.forward, grid=env.grid
            )
            env.render(particles=particles)
            time.sleep(0.1)

        env.step(Actions.right)
        particles = transition_model.sample(
            particles=particles, action=Actions.right, grid=env.grid
        )
        env.render(particles=particles)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
