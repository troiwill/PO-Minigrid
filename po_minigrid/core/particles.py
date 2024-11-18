from __future__ import annotations
from functools import cached_property
from typing import Any

from minigrid.core.world_object import WorldObj
import numpy as np


class Particles:

    def __init__(self, particles: np.ndarray, average_fn_name: str = "mean") -> None:
        # Sanity checks.
        assert average_fn_name in ("mean", "median")
        assert particles.dtype == Particles.dtype()

        self._values: np.ndarray
        self.data = particles

        self._average_fn_name = average_fn_name
        self._average_fn = np.mean if self._average_fn_name == "mean" else np.median

    @property
    def average(self) -> np.ndarray:
        state = np.round(self._average_fn(self.pose, axis=0), decimals=0).astype(int)
        return state

    @property
    def average_func_name(self) -> str:
        return self._average_fn_name

    @property
    def data(self) -> np.ndarray:
        return self._values

    @data.setter
    def data(self, values: np.ndarray) -> None:
        if not hasattr(self, "_values"):
            if values.dtype != Particles.dtype():
                raise TypeError(f"Type must be {Particles.dtype()}")
        elif self._values is not None:
            if values.shape != self._values.shape:
                raise ValueError("The shapes must be the same.")
            if values.dtype != self._values.dtype:
                raise ValueError("The dtypes must be the same.")
        self._values = values.copy()

    @property
    def carrying(self) -> np.ndarray:
        return self._values["carrying"]

    @carrying.setter
    def carrying(self, carrying: np.ndarray) -> None:
        _carrying = self.carrying
        if carrying.shape != _carrying.shape:
            raise ValueError(
                f"carrying must have shape {_carrying.shape}. Received {_carrying.shape}."
            )
        if carrying.dtype != _carrying.dtype:
            raise ValueError(
                f"carrying must have dtype {_carrying.dtype}. Received {_carrying.dtype}."
            )
        self._values["carrying"] = carrying.copy()

    @property
    def direction(self) -> np.ndarray:
        return self.pose[:, 2]

    @direction.setter
    def direction(self, direction: np.ndarray | None) -> None:
        if direction is None:
            direction = np.array([-1])

        _direction = self.direction
        if direction.shape != _direction.shape:
            raise ValueError(
                f"direction must have shape {_direction.shape}. Received {_direction.shape}."
            )
        if direction.dtype != _direction.dtype:
            raise ValueError(
                f"direction must have dtype {_direction.dtype}. Received {_direction.dtype}."
            )
        self._values["pose"][:, 2] = direction.copy()

    @property
    def pose(self) -> np.ndarray:
        return self._values["pose"]

    @pose.setter
    def pose(self, pose: np.ndarray) -> None:
        _pose = self.pose
        pose = pose.reshape(-1, 3)
        if pose.shape != _pose.shape:
            raise ValueError(
                f"pose must have shape {_pose.shape}. Received {_pose.shape}."
            )
        if pose.dtype != _pose.dtype:
            raise ValueError(
                f"pose must have dtype {_pose.dtype}. Received {_pose.dtype}."
            )
        self._values["pose"] = pose.copy()

    @property
    def position(self) -> np.ndarray:
        return self._values["pose"][:, :2]

    @position.setter
    def position(self, position: np.ndarray | None) -> None:
        if position is None:
            position = np.array([-1, -1])

        _position = self.position
        position = position.reshape(-1, 2)
        if position.shape != _position.shape:
            raise ValueError(
                f"position must have shape {_position.shape}. Received {position.shape}."
            )
        if position.dtype != _position.dtype:
            raise ValueError(
                f"position must have dtype {_position.dtype}. Received {position.dtype}."
            )
        self._values["pose"][:, :2] = position.copy()

    def __hash__(self) -> int:
        return hash(self._values.data.tobytes())

    def __getitem__(self, i: int) -> np.ndarray:
        return self.data[i]

    def __len__(self) -> int:
        return self._values.shape[0]

    def __setitem__(self, i: int, item: np.ndarray) -> None:
        self.data[i] = item

    @staticmethod
    def dtype() -> list[tuple[str, type] | tuple[str, type, tuple[Any]]]:
        return [("pose", np.int64, (3,)), ("carrying", np.object_)]

    @staticmethod
    def init(
        poses: np.ndarray | None = None,
        carrying: np.ndarray | None = None,
        average_fn_name: str = "mean",
    ) -> Particles:
        if isinstance(poses, np.ndarray) and isinstance(carrying, np.ndarray):
            assert len(poses) == len(carrying)

        if poses is None:
            poses = np.array([[-1, -1, 0]])

        init_array = np.zeros((len(poses),), dtype=Particles.dtype())
        init_array["pose"] = poses
        init_array["carrying"] = carrying
        return Particles(init_array, average_fn_name)

    def reset(self) -> None:
        self.pose = np.zeros_like(self.pose) - 1
        self.carrying = np.full_like(self.carrying, None)

    def unique_poses(self) -> np.ndarray:
        return np.unique(self.pose, axis=0)
