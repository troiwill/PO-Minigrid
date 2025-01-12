from __future__ import annotations
from typing import Protocol

import numpy as np

from po_minigrid.core.particles import Particles


class SampleBasedModel(Protocol):

    def sample(self, *args, **kwargs) -> Particles | np.ndarray:
        ...
