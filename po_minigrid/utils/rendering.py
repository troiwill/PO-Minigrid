from __future__ import annotations
from enum import IntEnum
import math
from typing import Callable

from minigrid.utils.rendering import (
    point_in_circle,
    point_in_triangle,
    rotate_fn,
)


class RENDER_OPTIONS(IntEnum):
    ARROW = 0
    CELL = 1


class AgentRender:

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_agent_rendering_fn(heading: int | None = None) -> Callable:
        # Sanity checks.
        assert heading is None or heading in (0, 1, 2, 3)

        # Create a render function for the true/estimated agent with direction.
        if heading is not None:
            true_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )
            return rotate_fn(true_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * heading)

        # Create a render function for the true/estimated agent without direction.
        else:
            return point_in_circle(0.5, 0.5, 0.375)

    @staticmethod
    def get_particle_rendering_fn(heading: int | None = None) -> Callable:
        # Sanity checks.
        assert heading is None or heading in (0, 1, 2, 3)

        # Create a render function for a particle with the agent's belief with direction.
        if heading is not None:
            particle_fn = point_in_triangle(
                (0.22, 0.29),
                (0.77, 0.50),
                (0.22, 0.71),
            )
            return rotate_fn(particle_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * heading)

        # Create a render function for a particle with the agent's belief without direction.
        else:
            return point_in_circle(0.5, 0.5, 0.275)

    @staticmethod
    def true_agent_color() -> tuple[int, int, int]:
        return (255, 0, 0)

    @staticmethod
    def estd_agent_color() -> tuple[int, int, int]:
        return (0, 255, 0)

    @staticmethod
    def particle_color() -> tuple[int, int, int]:
        return (173, 216, 230)
