from pycbc.detector import Detector
from bilby.gw.detector.networks import InterferometerList, get_empty_interferometer


ET1 = Detector('E1')
ET2 = Detector('E2')
ET3 = Detector('E3')

def  pycbc_antenna_response(det_name, pycbc_det, ra, dec, time, psi, mode):
    if mode == 'plus' or mode == 'cross':
        Fp, Fc = pycbc_det.antenna_pattern(right_ascension=ra,
                                     declination=dec,
                                     polarization=psi,
                                     t_gps=time,
                                     polarization_type='tensor')
        if mode == 'plus':
            return Fp
        else:
            return Fc
    elif mode == 'x' or mode =='y':
        Fx, Fy = pycbc_det.antenna_pattern(right_ascension=ra,
                                     declination=dec,
                                     polarization=psi,
                                     t_gps=time,
                                     polarization_type='vector')
        if mode == 'x':
            return Fx
        else:
            return Fy
    elif mode == 'breathing' or mode == 'longitudinal':
        Fb, Fl = pycbc_det.antenna_pattern(right_ascension=ra,
                                     declination=dec,
                                     polarization=psi,
                                     t_gps=time,
                                     polarization_type='scalar')
        if mode == 'breathing':
            return Fb
        else:
            return Fl
    elif mode == det_name:
        return 1.
    else:
        return 0.
        
class ET(InterferometerList):
    def __new__(cls):
        instance  = get_empty_interferometer('ET')
        # Overwrite the functions that compute the antenna patterns.
        instance[0].antenna_response = lambda ra, dec, time, psi, mode: pycbc_antenna_response('ET1', ET1, ra, dec, time, psi, mode)
        instance[1].antenna_response = lambda ra, dec, time, psi, mode: pycbc_antenna_response('ET2', ET2, ra, dec, time, psi, mode)
        instance[2].antenna_response = lambda ra, dec, time, psi, mode: pycbc_antenna_response('ET3', ET3, ra, dec, time, psi, mode)
        return instance