from __future__ import annotations

from minigrid.core.constants import DIR_TO_VEC
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj
import numpy as np

from po_minigrid.utils.decorate import np_vectorize


@np_vectorize(signature="()->()")
def can_enter_cell(cell: WorldObj | None) -> bool:
    """Determines if an agent can enter a given cell.

    This function checks whether a cell is empty or can be overlapped by an agent.

    Args:
        cell: A WorldObj instance representing the cell to check, or None if the cell is empty.

    Returns:
        bool: True if the cell is empty or can be overlapped, False otherwise.

    Notes:
        - This function is vectorized using numpy's vectorize decorator, allowing it to be
          applied efficiently to arrays of cells.
        - The signature "()->()" indicates that the function takes a scalar input and returns a scalar output.
    """
    return True if cell is None or cell.can_overlap() else False


@np_vectorize(signature="(n)->(m)")
def compute_fwd_pos(pose: np.ndarray) -> np.ndarray:
    """Computes the forward position based on the current pose.

    This function calculates the next position one step forward from the given pose.

    Args:
        pose: A numpy array of shape (3,) representing the current pose [x, y, theta].
              theta is expected to be an integer index corresponding to a direction in DIR_TO_VEC.

    Returns:
        position: A numpy array of shape (2,) representing the computed forward position [x, y].

    Notes:
        - This function is vectorized using numpy's vectorize decorator, allowing it to be
          applied efficiently to arrays of poses.
        - The signature "(n)->(m)" indicates that the function takes an n-dimensional input
          and returns an m-dimensional output.
        - DIR_TO_VEC is assumed to be a predefined constant mapping direction indices to vector offsets.
    """
    pos, theta = pose[:2], pose[2]
    dir_vec: np.ndarray = DIR_TO_VEC[theta].flatten()
    return pos + dir_vec


@np_vectorize(signature="(),(n)->()")
def get_grid_cell(grid: Grid, position: np.ndarray) -> WorldObj | None:
    """Computes the cell object in front of a given position on the grid.

    Args:
        grid: The Grid object representing the environment.
        position: A numpy array of shape (2,) representing the (x, y) position.

    Returns:
        The WorldObj at the given position, or None if the position is empty.
    """
    return grid.get(position[0], position[1])


def headings_are_valid(thetas: np.ndarray) -> bool:
    """Checks if all headings angles are valid (between 0 and 3 inclusive).

    Args:
        thetas: A numpy array of headings angles.

    Returns:
        True if all headings are valid, False otherwise.
    """
    return bool(np.all(0 <= thetas) and np.all(thetas < 4))
