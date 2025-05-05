from __future__ import annotations

import numpy as np
from numba import njit


@njit
def median_psd(time_frequency_data,
               time_frequency_filter):
    tlen, flen = time_frequency_data.shape
    output = np.zeros(flen)
    frequency_filter = np.zeros(flen).astype(bool)
    for j in range(flen):
        for i in range(tlen):
            if time_frequency_filter[i,j]:
                frequency_filter[j] = True
                spectrum = []
                for k in range(tlen):
                    if not time_frequency_filter[k,j]:
                        spectrum.append(time_frequency_data[k,j])
                if len(spectrum) > 0:
                    output[j] = np.median(spectrum)
                break
    return output
