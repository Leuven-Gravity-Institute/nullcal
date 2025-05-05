from __future__ import annotations

import numpy as np


def constant_calibration_model(frequency_array, constant):
    return np.full(len(frequency_array), constant)
