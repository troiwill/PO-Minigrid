from __future__ import annotations
from typing import Protocol

import numpy as np


class SampleBasedModel(Protocol):

    def sample(self, *args, **kwargs) -> np.ndarray: ...
