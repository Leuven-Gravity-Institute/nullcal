import unittest
import numpy as np
from nullcal.likelihood.projector import compute_calibrated_whitened_antenna_response
from nullcal.likelihood.projector import compute_projector
from nullcal.likelihood.projector import compute_projected_strain_data


class TestSelfRecalibrationProjectorLikelihood(unittest.TestCase):
    def test(self):
        nfreq = 4096
        ndet = 3
        nmode = 2
        whitened_antenna_response = np.random.randn(nfreq,ndet,nmode)+np.random.randn(nfreq,ndet,nmode)*1.j
        calibration_factor = np.random.randn(ndet,nfreq)+np.random.randn(ndet,nfreq)*1.j
        Fwc = compute_calibrated_whitened_antenna_response(whitened_antenna_response, calibration_factor)
        Fwc_np = np.einsum('fdm,df->fdm', whitened_antenna_response, calibration_factor)
        self.assertTrue(np.allclose(Fwc, Fwc_np))

    def test_compute_projector(self):
        nfreq = 4096
        ndet = 3
        nmode = 2
        calibrated_whitened_antenna_response_function = np.random.randn(nfreq,ndet,nmode)+np.random.randn(nfreq,ndet,nmode)*1.j
        projector = compute_projector(calibrated_whitened_antenna_response_function)
        # Compute the result with navie implmentation
        F_T = np.conj(np.transpose(calibrated_whitened_antenna_response_function, [0,2,1]))
        projector_np = np.array([np.eye(ndet) for i in range(nfreq)]) - calibrated_whitened_antenna_response_function @ np.linalg.inv(F_T @ calibrated_whitened_antenna_response_function) @ F_T
        self.assertTrue(np.allclose(projector, projector_np))

    def test_compute_projected_strain_data(self):
        nfreq = 4096
        ndet = 3
        nmode = 2
        calibrated_whitened_antenna_response_function = np.random.randn(nfreq,ndet,nmode)+np.random.randn(nfreq,ndet,nmode)*1.j
        projector = compute_projector(calibrated_whitened_antenna_response_function)
        strain_data = np.random.randn(nfreq, ndet) + np.random.randn(nfreq, ndet) * 1.j
        projected_strain_data = compute_projected_strain_data(projector, strain_data)
        projected_strain_data_np = np.einsum('fdl,fl->fd', projector, strain_data)
        self.assertTrue(np.allclose(projected_strain_data, projected_strain_data_np))

if __name__ == '__main__':
    seed = 12
    np.random.seed(seed)
    unittest.main()