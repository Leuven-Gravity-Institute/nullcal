from __future__ import annotations

import numpy as np
import scipy.stats
from bilby.core.prior.base import Prior
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


class RatioGaussian(Prior):
    def __init__(self, mu_x, sigma_x, mu_y, sigma_y, minimum, maximum, name=None, latex_label=None, unit=None, boundary=None):
        """A ratio Gaussian distribution of X/Y where X and Y are Gaussian random variables.

        Args:
            mu_x (float): Mean of X.
            sigma_x (float): Standard deviation of X.
            mu_y (float): Mean of Y.
            sigma_y (float): Standard deviation of Y.
            name (str, optional): Name of the parameter. Defaults to None.
            latex_label (str, optional): Latex label of the parameter. Defaults to None.
            unit (str, optional): Unit of the parameter. Defaults to None.
            boundary (str, optional): Boundary condition of the parameter. Defaults to None.
        """
        super().__init__(name=name, latex_label=latex_label, unit=unit, boundary=boundary)
        self.mu_x = mu_x
        self.sigma_x = sigma_x
        self.mu_y = mu_y
        self.sigma_y = sigma_y
        self._c = self.mu_x * self.mu_x / self.sigma_x / self.sigma_x + self.mu_y * self.mu_y / self.sigma_y / self.sigma_y
        self.minimum = minimum
        self.maximum = maximum
        # Compute the normalization constant
        x = np.linspace(self.minimum, self.maximum, 1000)
        y = self._unnormalized_prob(x)
        self.norm = np.trapz(y, x)
        # Construct the interpolator for the inverse cumulative distribution
        Y = cumulative_trapezoid(y/self.norm, x, initial=0)
        Y[-1] = 1.
        self.cumulative_distribution = interp1d(x=x, y=Y, bounds_error=False, fill_value=(0, 1))
        self.inverse_cumulative_distribution = interp1d(x=Y, y=x, bounds_error=True)

    def _unnormalized_prob(self, val):
        """Unormalized probability density.

        Args:
            val (float): Value.

        Returns:
            float: Unnormalized probabiltiy density.
        """
        a_z = np.sqrt(1. / self.sigma_x / self.sigma_x * val * val + 1. / self.sigma_y / self.sigma_y)
        b_z = self.mu_x / self.sigma_x / self.sigma_x * val + self.mu_y / self.sigma_y / self.sigma_y
        d_z = np.exp((b_z * b_z - self._c * a_z * a_z) / 2 / a_z / a_z)
        prob = b_z * d_z / a_z ** 3 / np.sqrt(2 * np.pi) / self.sigma_x / self.sigma_y * (scipy.stats.norm.cdf(b_z / a_z) - scipy.stats.norm.cdf(-b_z / a_z)) + np.exp(-self._c / 2) / a_z / a_z / np.pi / self.sigma_x / self.sigma_y
        return prob

    def prob(self, val):
        """Probability density.

        Args:
            val (float): Value.

        Returns:
            float: Probability density.
        """
        return self._unnormalized_prob(val) / self.norm

    def cdf(self, val):
        """Cumulative distribution.

        Args:
            val (float): Value.

        Returns:
            float: Cumulative distribution.
        """
        x = np.linspace(self.minimum, val, 1000)
        y = self.prob(x)
        return np.trapz(y, x)

    def rescale(self, val):
        """Rescale a uniform random variable to the ratio Gaussian distribution.

        Args:
            val (float): Value.

        Returns:
            float: Value.
        """
        rescaled = self.inverse_cumulative_distribution(val)
        if rescaled.shape == ():
            rescaled = float(rescaled)
        return rescaled
