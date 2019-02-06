import numpy as np
from scipy import optimize

""" 
Define a bunch of smaller functions I need to get the radial velocity values

_gp and _hp need the same arguments as _g and _h in order to work with newton
"""


def _mean_anom(date, p, phi):
    return 2 * np.pi * date / p - phi

def _g(e, m, ec):
    return e - ec * np.sin(e) - m

def _gp(e, m, ec):
    return 1 - ec * np.cos(e)

def _h(f, e, ec):
    return np.cos(f) - (np.cos(e) - ec) / (1 - ec * np.cos(e))

def _hp(f, e, ec):
    return -1 * np.sin(f)

def rv_from_param(v0 , k, period, ec, phi, omega, date):
    """
    Finds the radial velocity along our line of sight from some of the orbital paramiters
    :param v0: Barrycenter velocity
    :param k: Don't remember the name of this, but it's like the velocity in it's refference frame
    :param period: period of the orbit
    :param ec: eccentricity of the orbit
    :param phi:
    :param omega:
    :param date: times I want to know the radial velocity at
    :return rv: Radial velocity in our line of sight for the input dates
    """
    M = _mean_anom(date, period, phi)

    E0 = M.copy()
    E = optimize.newton(_g, E0, fprime = _gp, args = (M, ec))

    f0 = E.copy()
    f = optimize.newton(_h, f0, fprime = _hp, args = (E, ec))

    in_frame_rv = np.cos(omega + f) + ec*np.cos(omega)
    rv = k * in_frame_rv
    rv += v0

    return rv


