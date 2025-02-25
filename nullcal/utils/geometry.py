import numpy as np


def get_vertex_position_ellipsoid(x_comp, y_comp, z_comp):
    """
    Calculate the position of the IFO vertex in ellipsoidal coordinates.

    Implementation of Borkowski's algorithm for the exact inverse of get_vertex_position_geocentric().
    See http://www.astro.uni.torun.pl/~kb/Papers/ASS/Geod-ASS.htm for details.
    For the correct version of formula 11a, see: http://www.astro.uni.torun.pl/~kb/Papers/geod/Geod-GK.htm#GEOD
    See https://ieeexplore.ieee.org/document/303772 for an overview of inversion algorithms.

    Parameters
    ==========
    x_comp: float
        Geocentric x-coordinate in meters
    y_comp: float
        Geocentric y-coordinate in meters
    z_comp: float
        Geocentric z-coordinate in meters

    Returns
    =======
    array_like: A 3D representation of the ellipsoidal vertex position (latitude [rad], longitude [rad], elevation [m])
    """
    semi_major_axis = 6378137  # for ellipsoid model of Earth, in m
    semi_minor_axis = 6356752.314  # in m

    r = np.sqrt(x_comp ** 2 + y_comp ** 2)
    E = (semi_minor_axis * z_comp - (semi_major_axis ** 2 - semi_minor_axis ** 2)) / (r * semi_major_axis)
    F = (semi_minor_axis * z_comp + (semi_major_axis ** 2 - semi_minor_axis ** 2)) / (r * semi_major_axis)
    P = 4 / 3 * (E * F + 1)
    Q = 2 * (E ** 2 - F ** 2)
    D = P ** 3 + Q ** 2
    v = (np.sqrt(D) - Q) ** (1 / 3) - (np.sqrt(D) + Q) ** (1 / 3)

    # Calculate solution in first quadrant and then adjust based on the sign of the original z_comp
    G = 1 / 2 * (np.sqrt(E ** 2 + v) + E)
    t = np.sqrt(G ** 2 + (F - v * G) / (2 * G - E)) - G
    latitude = np.sign(z_comp) * np.arctan((semi_major_axis * (1 - t ** 2)) / (2 * semi_minor_axis * t))
    longitude = np.arctan2(y_comp, x_comp)
    elevation = (r - semi_major_axis * t) * np.cos(latitude) + (z_comp - semi_minor_axis) * np.sin(latitude)

    return np.array([latitude, longitude, elevation])
