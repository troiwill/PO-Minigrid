from __future__ import annotations

import copy
from typing import Any, SupportsFloat

from gymnasium.core import ActType, ObsType
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import TILE_PIXELS
from minigrid.core.mission import MissionSpace
import numpy as np
import numpy.typing as npt
import pygame

from po_minigrid.core.po_grid import POGrid


class POMiniGridEnv(MiniGridEnv):
    """
    2D grid world game environment with partial observability.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        mission_space: MissionSpace,
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
        num_particles: int = 100,
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
        )
        # Reset the grid as a PO Grid.
        self.grid = POGrid(self.width, self.height)
        self.step_count = 0

        # Reset the observation space if the parameter is not None.
        if observation_space is not None:
            self.observation_space = copy.deepcopy(observation_space)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # Avoid rendering now if it was set.
        old_render_mode = self.render_mode
        self.render_mode = "rgb_array"

        # Call super().render()
        obs, reward, terminated, truncated, aux_info = super().step(action=action)

        # Reset the render mode.
        self.render_mode = old_render_mode

        return obs, reward, terminated, truncated, aux_info

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

        for i in range(self.agent_dir + 1):
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
        print(type(grid))
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
        particles: npt.ArrayLike | None = None,
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

    def render(self, particles: npt.ArrayLike | None = None) -> np.ndarray:
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
