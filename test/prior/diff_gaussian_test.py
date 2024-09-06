import unittest
import bilby
import scipy.stats
import nullcal.prior


class TestDiffGaussian(unittest.TestCase):
    def test_sample(self):
        seed = 1
        bilby.core.utils.random.seed(seed)
        nsample = 1000
        mu_x = 1
        sigma_x = 1
        mu_y = 3
        sigma_y = 2
        prior = nullcal.prior.diff_gaussian.DiffGaussian(mu_x=mu_x, sigma_x=sigma_x, mu_y=mu_y, sigma_y=sigma_y)
        gaussian_x = bilby.core.prior.Gaussian(mu=mu_x, sigma=sigma_x)
        gaussian_y = bilby.core.prior.Gaussian(mu=mu_y, sigma=sigma_y)
        samples = prior.sample(nsample)
        true_distribution = gaussian_x.sample(nsample) - gaussian_y.sample(nsample)
        result = scipy.stats.ks_2samp(samples, true_distribution)
        self.assertGreaterEqual(result.pvalue, 0.05)

if __name__ == '__main__':
    unittest.main()        