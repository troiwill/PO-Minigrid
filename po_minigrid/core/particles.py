from __future__ import annotations

import copy
from functools import cached_property
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
from minigrid.core.world_object import WorldObj

from po_minigrid.models.utils import compute_fwd_pos

ParticleArrayDtype = List[Union[Tuple[str, type], Tuple[str, type, Tuple[Any]]]]
ParticleArrayIndex = Union[int, Sequence[int], np.ndarray]
ParticleArrayObjectValues = Union[
    WorldObj, None, np.ndarray, Sequence[Union[WorldObj, None, Any]]
]


class WorldObjectArray:

    def __init__(self, length: int, is_belief: bool) -> None:
        self._data = np.zeros((length,), dtype=np.object_)
        self._is_belief = is_belief
        self.reset()

    def __len__(self) -> int:
        return self._data.shape[0]

    def reset(self) -> None:
        if self._is_belief:
            self._data = np.array(list(dict() for _ in range(len(self))))
        else:
            self._data = np.array(list(None for _ in range(len(self))))

    def get(
        self, indices: None | ParticleArrayIndex = None
    ) -> WorldObj | None | np.ndarray:
        if indices is None:
            return self._data
        else:
            return self._data[indices]

    def set(
        self,
        values: ParticleArrayObjectValues,
        indices: None | ParticleArrayIndex = None,
    ) -> None:
        if indices is None:
            indices = range(len(self._data))
        elif isinstance(indices, int):
            indices = [indices]
        assert len(set(indices)) == len(indices)

        if isinstance(values, WorldObj):
            values = [values]
        elif values is None:
            values = [None] * len(indices)

        assert len(indices) == len(values)
        if self._is_belief:
            copied_values = [dict() if v is None else copy.deepcopy(v) for v in values]
        else:
            copied_values = values

        for i, v, c in zip(indices, values, copied_values):
            if v is None:
                self._data[i] = c
            else:
                self._data[i][id(v)] = c

    def __str__(self) -> str:
        return str(self._data)


class Particles:

    def __init__(
        self,
        pose: np.ndarray | None = None,
        is_belief: bool = True,
        average_fn_name: str = "mean",
    ) -> None:
        # Sanity checks.
        assert isinstance(pose, np.ndarray) or pose is None
        if isinstance(pose, np.ndarray):
            assert len(pose.shape) == 2 and pose.shape[0] >= 1 and pose.shape[1] == 3
        assert isinstance(is_belief, bool)
        assert average_fn_name in ("mean", "median")

        # Set the is belief attribute.
        self._is_belief = is_belief

        # Initialize the pose, carrying, and toggled arrays.
        self._reset_pose = np.array([[-1, -1, 0]], dtype=np.int64)
        if pose is None:
            pose = self._reset_pose.copy()
        pose = pose.reshape(-1, 3).astype(np.int64)
        self._pose = pose.copy()

        pose_len = len(pose)
        self._carrying = WorldObjectArray(length=pose_len, is_belief=self._is_belief)
        self._toggled = WorldObjectArray(length=pose_len, is_belief=self._is_belief)
        self._average_fn_name = average_fn_name

    def __len__(self) -> int:
        return len(self._pose)

    @cached_property
    def is_belief(self) -> bool:
        return self._is_belief

    @property
    def carrying(self) -> WorldObjectArray:
        return self._carrying

    @property
    def toggled(self) -> WorldObjectArray:
        return self._toggled

    @property
    def fwd_pos(self) -> np.ndarray:
        return compute_fwd_pos(self.pose)

    @property
    def pose(self) -> np.ndarray:
        return self._pose

    @pose.setter
    def pose(self, pose: np.ndarray | None) -> None:
        if pose is None:
            pose = np.repeat(self._reset_pose, len(self), axis=0)
        assert self._pose.shape == pose.shape
        assert self._pose.dtype == pose.dtype
        self._pose = pose.copy()

    @property
    def position(self) -> np.ndarray:
        return self.pose[:, :2]

    @position.setter
    def position(self, position: np.ndarray | None) -> None:
        if position is None:
            position = np.repeat(self._reset_pose[:, :2], len(self), axis=0)
        assert self._pose[:, :2].shape == position.shape
        assert self._pose.dtype == position.dtype
        self._pose[:, :2] = position.copy()

    @property
    def direction(self) -> np.ndarray:
        return self.pose[:, 2]

    @direction.setter
    def direction(self, direction: np.ndarray | None) -> None:
        if direction is None:
            direction = np.repeat(self._reset_pose[:, 2], len(self), axis=0)
        assert self._pose[:, 2].shape == direction.shape
        assert self._pose.dtype == direction.dtype
        self._pose[:, 2] = direction.copy()

    @property
    def average(self) -> np.ndarray:
        if self._average_fn_name == "mean":
            return self.mean
        else:
            return self.median

    @property
    def mean(self) -> np.ndarray:
        return np.round(np.mean(self.pose, axis=0), decimals=0).astype(int)

    @property
    def median(self) -> np.ndarray:
        return np.round(np.median(self.pose, axis=0), decimals=0).astype(int)

    def reset(self) -> None:
        self.pose = None
        self.carrying.reset()
        self.toggled.reset()

    def unique_poses(self) -> np.ndarray:
        return np.unique(self.pose, axis=0)
