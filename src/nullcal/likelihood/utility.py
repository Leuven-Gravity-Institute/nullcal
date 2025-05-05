from __future__ import annotations

import numpy as np
from numba import njit


@njit
def compute_frequency_mask(time_frequency_filter):
    tlen, flen = time_frequency_filter.shape
    output = np.zeros(flen).astype(bool)
    for j in range(flen):
        for i in range(tlen):
            if time_frequency_filter[i,j]:
                output[j] = True
                break
    return output
