import unittest
import bilby
import scipy.stats
import nullcal.prior


class TestRatioGaussian(unittest.TestCase):
    def test_sample(self):
        seed = 1
        bilby.core.utils.random.seed(seed)
        nsample = 1000
        mu_x = 1
        sigma_x = 1
        mu_y = 3
        sigma_y = 2
        minimum = -10
        maximum = 10
        prior = nullcal.prior.ratio_gaussian.RatioGaussian(mu_x=mu_x, sigma_x=sigma_x, mu_y=mu_y, sigma_y=sigma_y, minimum=minimum, maximum=maximum)
        gaussian_x = bilby.core.prior.Gaussian(mu=mu_x, sigma=sigma_x)
        gaussian_y = bilby.core.prior.Gaussian(mu=mu_y, sigma=sigma_y)
        samples = prior.sample(nsample)
        true_distribution = []
        while len(true_distribution) < nsample:
            sample = gaussian_x.sample() / gaussian_y.sample()
            if sample >= minimum and sample <= maximum:
                true_distribution.append(sample)
        result = scipy.stats.ks_2samp(samples, true_distribution)
        self.assertGreaterEqual(result.pvalue, 0.05)
        
if __name__ == '__main__':
    unittest.main()