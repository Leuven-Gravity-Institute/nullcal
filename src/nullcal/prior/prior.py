from __future__ import annotations

import bilby
import numpy as np
from bilby.gw.prior import CalibrationPriorDict as BilbyCalibrationPriorDict


class CalibrationPriorDict(BilbyCalibrationPriorDict):
    def __init__(self, dictionary=None, filename=None):
        super.__init__(dictionary=dictionary, filename=filename)

    @staticmethod
    def from_result_file(result_file,
                         minimum_frequency,
                         maximum_frequency,
                         n_nodes,
                         boundary="reflective"):
        result = bilby.core.result.read_in_result(result_file)
        mus = np.mean(result.sample, axis=0)
        covs = np.cov(result.sample.T)
