from __future__ import annotations

import numpy as np


def np_vectorize(**kwargs):
    def decorator(func):
        return np.vectorize(func, **kwargs)

    return decorator
