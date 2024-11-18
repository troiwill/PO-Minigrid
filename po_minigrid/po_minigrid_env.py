from __future__ import annotations

import copy
from typing import Any, SupportsFloat

from gymnasium.core import ActType, ObsType
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import TILE_PIXELS
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj
import numpy as np
import pygame

from po_minigrid.core.particles import Particles
from po_minigrid.core.po_grid import POGrid
from po_minigrid.models import SampleBasedModel
from po_minigrid.models.reward import RewardModel
from po_minigrid.models.observation import ObservationModel
from po_minigrid.models.transition import TransitionModel


class POMiniGridEnv(MiniGridEnv):
    """
    2D grid world game environment with partial observability.
    """

    def __init__(
        self,
        mission_space: MissionSpace,
        transition_model: SampleBasedModel | None = None,
        observation_model: SampleBasedModel | None = None,
        reward_model: SampleBasedModel | None = None,
        grid_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
        max_steps: int = 100,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        render_mode: str | None = None,
        screen_size: int | None = 640,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
        observation_space: Any | None = None,
        **kwargs,
    ):
        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            agent_view_size=agent_view_size,
            render_mode=render_mode,
            screen_size=screen_size,
            highlight=highlight,
            tile_size=tile_size,
            agent_pov=agent_pov,
            **kwargs,
        )
        # Reset the grid as a PO Grid.
        self.grid = POGrid(self.width, self.height)
        self.step_count = 0
        self._terminal_states = None

        # Create the state variable.
        self._agent_state: np.ndarray = np.array([])

        # Create the models.
        self.transition_model: SampleBasedModel = (
            transition_model if transition_model is not None else TransitionModel()
        )
        self.observation_model: SampleBasedModel = (
            observation_model if observation_model is not None else ObservationModel()
        )
        self.reward_model: SampleBasedModel = (
            reward_model if reward_model is not None else RewardModel()
        )

        # Reset the observation space if the parameter is not None.
        if observation_space is not None:
            self.observation_space = copy.deepcopy(observation_space)

    # @staticmethod
    # def _cell_is_in_terminal_state(cell: WorldObj | None) -> bool:
    #     return False if cell is None or cell.type not in ("lava", "goal") else True

    @property
    def agent_pose(self) -> np.ndarray:
        (x, y), theta = self.agent_pos, self.agent_dir
        return np.array([x, y, theta])

    @property
    def agent_state(self) -> np.ndarray:
        raise

    def has_exceed_max_steps(self) -> bool:
        return self.step_count >= self.max_steps
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        # Reinitialize episode-specific variables
        self.agent_pos = (-1, -1)
        self.agent_dir = -1

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert (
            self.agent_pos >= (0, 0)
            if isinstance(self.agent_pos, tuple)
            else all(self.agent_pos >= 0) and self.agent_dir >= 0
        )

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        return obs, {}

    def step(
        self,
        action: ActType,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Apply the transition model.
        next_pose = (
            self.transition_model.sample(
                states=self.agent_pose, action=action, grid=self.grid
            )
            .flatten()
            .tolist()
        )
        self.agent_pos = (next_pose[0], next_pose[1])
        self.agent_dir = next_pose[2]

        next_cell = self.grid.get(*self.agent_pos)
        if next_cell is not None and next_cell.type == "goal":
            terminated = True
            reward = self.reward_model.sample(
                states=self.agent_state, action=action, grid=self.grid
            )
        if next_cell is not None and next_cell.type == "lava":
            terminated = True

        # # Pick up an object
        # elif action == self.actions.pickup:
        #     if fwd_cell and fwd_cell.can_pickup():
        #         if self.carrying is None:
        #             self.carrying = fwd_cell
        #             self.carrying.cur_pos = np.array([-1, -1])
        #             self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # # Drop an object
        # elif action == self.actions.drop:
        #     if not fwd_cell and self.carrying:
        #         self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
        #         self.carrying.cur_pos = fwd_pos
        #         self.carrying = None

        # # Toggle/activate an object
        # elif action == self.actions.toggle:
        #     if fwd_cell:
        #         fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        # elif action == self.actions.done:
        #     pass

        # else:
        #     raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}
    
    def _update_minigrid_vars(self) -> None:
        agent_state = self.agent_state
        self.agent_pos = (agent_state[0], agent_state[1])
        self.agent_dir = agent_state[2]

    def gen_obs_grid(self, agent_view_size=None):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """

        topX, topY, botX, botY = self.get_view_exts(agent_view_size)

        agent_view_size = agent_view_size or self.agent_view_size

        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

        for _ in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(
                agent_pos=(agent_view_size // 2, agent_view_size - 1)
            )
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask

    def get_pov_render(self, tile_size, particles):
        """
        Render an agent's POV observation for visualization
        """
        grid, vis_mask = self.gen_obs_grid()

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1),
            agent_dir=3,
            highlight_mask=vis_mask,
            particles=particles,
        )

        return img

    def get_full_render(self, highlight, tile_size, particles):
        """
        Render a non-paratial observation for visualization
        """
        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = (
            self.agent_pos
            + f_vec * (self.agent_view_size - 1)
            - r_vec * (self.agent_view_size // 2)
        )

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None,
            particles=particles,
        )

        return img

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
        particles: np.ndarray | None = None,
    ):
        """Returns an RGB image corresponding to the whole environment or the agent's point of view.

        Args:

            highlight (bool): If true, the agent's field of view or point of view is highlighted with a lighter gray color.
            tile_size (int): How many pixels will form a tile from the NxM grid.
            agent_pov (bool): If true, the rendered frame will only contain the point of view of the agent.

        Returns:
            frame (np.ndarray): A frame of type numpy.ndarray with shape (x, y, 3) representing RGB values for the x-by-y pixel image.

        """

        if agent_pov:
            return self.get_pov_render(tile_size, particles)
        else:
            return self.get_full_render(highlight, tile_size, particles)

    def render(self, particles: np.ndarray | None = None) -> np.ndarray:
        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov, particles)

        if self.render_mode == "human":
            img = np.transpose(img, axes=(1, 0, 2))
            if self.render_size is None:
                self.render_size = img.shape[:2]
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.screen_size, self.screen_size)
                )
                pygame.display.set_caption("minigrid")
            if self.clock is None:
                self.clock = pygame.time.Clock()
            surf = pygame.surfarray.make_surface(img)

            # Create background with mission description
            offset = surf.get_size()[0] * 0.1
            # offset = 32 if self.agent_pov else 64
            bg = pygame.Surface(
                (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
            )
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))

            bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

            font_size = 22
            text = self.mission
            font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
            text_rect = font.get_rect(text, size=font_size)
            text_rect.center = bg.get_rect().center
            text_rect.y = bg.get_height() - font_size * 1.5
            font.render_to(bg, text_rect, text, size=font_size)

            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return img

    def close(self):
        if self.window:
            pygame.quit()
