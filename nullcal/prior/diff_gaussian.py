import numpy as np
from bilby.core.prior.analytical import Gaussian


class DiffGaussian(Gaussian):
    def __init__(self, mu_x, sigma_x, mu_y, sigma_y, name=None, latex_label=None, unit=None, boundary=None):
        super().__init__(mu=mu_x-mu_y, sigma=np.sqrt(sigma_x * sigma_x + sigma_y * sigma_y), name=name, latex_label=latex_label, unit=unit, boundary=boundary)