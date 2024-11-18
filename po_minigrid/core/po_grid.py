from __future__ import annotations
from typing import Any

import numpy as np
from minigrid.core.constants import TILE_PIXELS, OBJECT_TO_IDX
from minigrid.core.world_object import Wall, WorldObj
from minigrid.utils.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_rect,
)
import minigrid.core.grid

from po_minigrid.core.particles import Particles
from po_minigrid.utils.rendering import AgentRender


class POGrid(minigrid.core.grid.Grid):
    """
    Represents a grid with partial observability and operations on it.
    """

    def slice(self, topX: int, topY: int, width: int, height: int) -> POGrid:
        """
        Get a subset of the grid.
        """
        grid = POGrid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if 0 <= x < self.width and 0 <= y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall()

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(
        cls,
        obj: WorldObj | None,
        agent_dir: int | None = None,
        highlight: bool = False,
        tile_size: int = TILE_PIXELS,
        subdivs: int = 3,
        belief_avg_dir: int | None = None,
        belief_dirs: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """
        Render a tile and cache the result.
        """

        # Hash map lookup key for the cache
        key: tuple[Any, ...] = (
            agent_dir,
            belief_avg_dir,
            hash(belief_dirs),
            highlight,
            tile_size,
        )
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            render_fn = AgentRender.get_agent_rendering_fn(
                None if agent_dir < 0 else agent_dir
            )
            fill_coords(img, render_fn, AgentRender.true_agent_color())

        # Overlay the agent mean on top
        if belief_avg_dir is not None:
            render_fn = AgentRender.get_agent_rendering_fn(
                None if belief_avg_dir < 0 else belief_avg_dir
            )
            fill_coords(img, render_fn, AgentRender.estd_agent_color())

        # Overlay the belief/particles on top
        if belief_dirs is not None:
            for pdir in belief_dirs:
                render_fn = AgentRender.get_particle_rendering_fn(
                    None if pdir < 0 else pdir
                )
                fill_coords(img, render_fn, AgentRender.particle_color())

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
        self,
        tile_size: int,
        agent_pos: tuple[int, int] | np.ndarray,
        agent_dir: int | None = None,
        highlight_mask: np.ndarray | None = None,
        particles: Particles | None = None,
    ) -> np.ndarray:
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Get all the unique states within the belief so that we do not draw multiple same states.
        particles_average: tuple[int, int, int] | None = None
        unique_particle_data: np.ndarray | None = None
        if particles is not None:
            particles_average = particles.average
            unique_particle_data = particles.unique()

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                curr_grid_pos = (i, j)
                cell = self.get(*curr_grid_pos)

                # Determines if the true agent is at this position.
                agent_here = np.array_equal(agent_pos, curr_grid_pos)

                # If the belief was passed, find the unique agent belief directions (headings).
                belief_dirs = None
                if unique_particle_data is not None:
                    position_match_indices = np.array(
                        [
                            np.array_equal(row, curr_grid_pos)
                            for row in unique_particle_data[:, :2]
                        ]
                    )

                    if np.any(position_match_indices):
                        belief_dirs = tuple(
                            unique_particle_data[position_match_indices, 2].tolist()
                        )

                belief_avg_dir = None
                if particles_average is not None and np.array_equal(
                    particles_average[:2], curr_grid_pos
                ):
                    belief_avg_dir = particles_average[2]

                assert highlight_mask is not None
                tile_img = POGrid.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                    belief_avg_dir=belief_avg_dir,
                    belief_dirs=belief_dirs,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    @staticmethod
    def decode(array: np.ndarray) -> tuple[POGrid, np.ndarray]:
        """
        Decode an array grid encoding back into a grid
        """

        width, height, channels = array.shape
        assert channels == 3

        vis_mask = np.ones(shape=(width, height), dtype=bool)

        grid = POGrid(width, height)
        for i in range(width):
            for j in range(height):
                type_idx, color_idx, state = array[i, j]
                v = WorldObj.decode(type_idx, color_idx, state)
                grid.set(i, j, v)
                vis_mask[i, j] = type_idx != OBJECT_TO_IDX["unseen"]

        return grid, vis_mask

    def __eq__(self, other: POGrid) -> bool:
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other: POGrid) -> bool:
        return not self == other

    def copy(self) -> POGrid:
        from copy import deepcopy

        return deepcopy(self)

    def rotate_left(self) -> POGrid:
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = POGrid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid
