from numba import njit, prange
import numpy as np

@njit
def compute_projector(calibrated_whitened_antenna_response_function, frequency_mask):
    """Compute the projector given the calibrated whitened antenna response function.

    Args:
        calibrated_whitened_antenna_response_function (numpy array): Calibrated whitened antenna response function.

    Returns:
        numpy array: Projector.
    """
    # The dimensions of the input is (frequency, detector, mode)
    nfreq, ndet, nmode = calibrated_whitened_antenna_response_function.shape
    FtF = np.zeros((nfreq, nmode, nmode), dtype=calibrated_whitened_antenna_response_function.dtype)
    for i in range(nfreq):
        if frequency_mask[i]:
            for j in range(nmode):
                for k in range(j, nmode):
                    sum = 0.
                    for l in range(ndet):
                        sum += np.conj(calibrated_whitened_antenna_response_function[i,l,j])*calibrated_whitened_antenna_response_function[i,l,k]
                    FtF[i,j,k] = sum
                    FtF[i,k,j] = np.conj(sum)
    # Compute F @ FtF_inv first
    F_FtF_inv = np.zeros((nfreq, ndet, nmode), dtype=calibrated_whitened_antenna_response_function.dtype)
    for i in range(nfreq):
        if frequency_mask[i]:
            FtF_inv_i = np.linalg.inv(FtF[i])
            for j in range(ndet):
                for k in range(nmode):
                    sum = 0.
                    for l in range(nmode):
                        sum += calibrated_whitened_antenna_response_function[i,j,l]*FtF_inv_i[l,k]
                    F_FtF_inv[i,j,k] = sum                
    output = np.zeros((nfreq, ndet, ndet), dtype=calibrated_whitened_antenna_response_function.dtype)
    for i in range(nfreq):
        if frequency_mask[i]:
            for j in range(ndet):
                for k in range(j, ndet):
                    sum = 0.
                    for l in range(nmode):
                        sum += F_FtF_inv[i,j,l]*np.conj(calibrated_whitened_antenna_response_function[i,k,l])
                    if j != k:
                        output[i,j,k] = -sum
                        output[i,k,j] = -np.conj(sum)
                    else:
                        output[i,j,k] = 1. - sum
    return output

def compute_projector_numpy(calibrated_whitened_antenna_response_function, frequency_mask):
    nfreq, ndet, nmode = calibrated_whitened_antenna_response_function.shape
    calibrated_whitened_antenna_response_function_masked = calibrated_whitened_antenna_response_function[frequency_mask,:,:]
    output = np.eye(ndet,dtype=np.complex128)[np.newaxis, :, :].repeat(nfreq, axis=0)
    F_dagger = np.transpose(np.conj(calibrated_whitened_antenna_response_function_masked), axes=(0,2,1))
    #I - F@(F'F)^-1 F'
    Pgw = np.matmul(calibrated_whitened_antenna_response_function_masked,np.matmul(np.linalg.inv(np.matmul(F_dagger,calibrated_whitened_antenna_response_function_masked)),F_dagger))
    output[frequency_mask,:,:] = output[frequency_mask,:,:] - Pgw
    return output

if __name__ == '__main__':
    nfreq = 2000000
    ndet = 3
    nmode = 2
    frequency_mask = np.ones(nfreq, dtype=int)
    frequency_mask[:1000] = 0
    frequency_mask = frequency_mask.astype(bool)
    np.random.seed(1)
    F = np.random.randn(nfreq, ndet, nmode) + np.random.randn(nfreq, ndet, nmode) * 1.j
    Pnull_1 = compute_projector(F, frequency_mask)
    Pnull_2 = compute_projector_numpy(F, frequency_mask)
    print('v1')
    print(Pnull_1[frequency_mask,:,:])
    print('v2')
    print(Pnull_2[frequency_mask,:,:])

    import timeit
    time_1 = timeit.timeit('compute_projector(F, frequency_mask)', 'from __main__ import compute_projector, F, frequency_mask', number=10)
    print('v1:', time_1 / 10)
    time_2 = timeit.timeit('compute_projector_numpy(F, frequency_mask)', 'from __main__ import compute_projector_numpy, F, frequency_mask', number=10)
    print('v2:', time_2 / 10)