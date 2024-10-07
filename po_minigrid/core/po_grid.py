from __future__ import annotations
import math
from typing import Any

import numpy as np
import numpy.typing as npt
from minigrid.core.constants import TILE_PIXELS, OBJECT_TO_IDX
from minigrid.core.world_object import Wall, WorldObj
from minigrid.utils.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)
import minigrid.core.grid


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
        particles_central_dir: int | None = None,
        particles_dirs: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """
        Render a tile and cache the result.
        """

        # Hash map lookup key for the cache
        key: tuple[Any, ...] = (
            agent_dir,
            particles_central_dir,
            particles_dirs,
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
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
            fill_coords(img, tri_fn, (255, 0, 0))

        # Overlay the agent mean on top
        if particles_central_dir is not None:
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(
                tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * particles_central_dir
            )
            fill_coords(img, tri_fn, (0, 255, 0))

        # Overlay the belief/particles on top
        if particles_dirs is not None:
            for dir in particles_dirs:
                tri_fn = point_in_triangle(
                    (0.22, 0.29),
                    (0.77, 0.50),
                    (0.22, 0.71),
                )

                # Rotate the agent based on its direction
                tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * dir)
                fill_coords(img, tri_fn, (173, 216, 230))

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
        agent_pos: tuple[int, int],
        agent_dir: int | None = None,
        highlight_mask: np.ndarray | None = None,
        particles: npt.ArrayLike | None = None,
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
        particles_average = None
        if particles is not None:
            particles = np.unique(np.array(particles, dtype=int), axis=0)
            particles_average = tuple(
                np.round(np.average(particles, axis=0), decimals=0).astype(int).tolist()
            )

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                curr_grid_pos = (i, j)
                cell = self.get(*curr_grid_pos)

                # Determines if the true agent is at this position.
                agent_here = np.array_equal(agent_pos, curr_grid_pos)

                # If the belief was passed, find the unique agent belief directions (headings).
                particles_dirs = None
                if particles is not None:
                    particles_pos = particles[:, :2]
                    position_match_indices = np.array(
                        [np.array_equal(row, curr_grid_pos) for row in particles_pos]
                    )

                    if np.any(position_match_indices):
                        particles_dirs = tuple(
                            particles[position_match_indices, 2].tolist()
                        )

                average_dir = None
                if particles_average is not None and np.array_equal(
                    particles_average[:2], curr_grid_pos
                ):
                    average_dir = particles_average[2]

                assert highlight_mask is not None
                tile_img = POGrid.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                    particles_central_dir=average_dir,
                    particles_dirs=particles_dirs,
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
