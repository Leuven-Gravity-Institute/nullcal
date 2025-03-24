from __future__ import annotations

import bilby
import numpy as np
from bilby.core.prior import MultivariateGaussianDist


class GaussianFit(MultivariateGaussianDist):
    def __init__(self, result_file):
        result = bilby.core.result.read_in_result(result_file)
        samples = result.samples
        super().__init__(names=list(result.posterior.columns[:-2]), mus=np.mean(samples,axis=0), covs=np.cov(samples.T))
