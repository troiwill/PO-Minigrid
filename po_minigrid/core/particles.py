from __future__ import annotations
from functools import cached_property

import numpy as np


class Particles:

    def __init__(self, particles: np.ndarray, average_fn_name: str = "mean") -> None:
        assert average_fn_name in ("mean", "median")
        self._values: np.ndarray = particles.astype(int)
        self._average_fn_name = average_fn_name
        self._average_fn = np.mean if self._average_fn_name == "mean" else np.median

    @property
    def average(self) -> tuple[int, int, int]:
        state = (
            np.round(self._average_fn(self._values, axis=0), decimals=0)
            .astype(int)
            .tolist()
        )
        return state[0], state[1], state[2]

    @property
    def average_func_name(self) -> str:
        return self._average_fn_name

    @property
    def data(self) -> np.ndarray:
        return self._values

    @data.setter
    def data(self, values: np.ndarray) -> None:
        if values.shape != self._values.shape:
            raise ValueError("The shapes must be the same.")
        if values.dtype != self._values.dtype:
            raise ValueError("The dtypes must be the same.")
        self._values = values.copy()

    # @cached_property
    # def dtype(self):
    #     return (("pose", "i4"), ("carrying", bool))

    # @property
    # def positions(self) -> np.ndarray:
        

    def __getitem__(self, i: int) -> np.ndarray:
        return self.data[i]

    def __len__(self) -> int:
        return self._values.shape[0]

    def __setitem__(
        self, i: int, item: np.ndarray | tuple[int, int, int] | list[int]
    ) -> None:
        self.data[i] = item

    def unique(self) -> np.ndarray:
        return np.unique(self._values, axis=0)

    def __hash__(self) -> int:
        return hash(self._values.data.tobytes())
