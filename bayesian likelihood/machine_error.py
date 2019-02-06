import numpy as np
import astropy.units as u

def machine_error(rv, error):
    """
    Adds an extra machine error to the rv values based on the size of the error. Used for making the synthetic rv signals

    :param rv:
    :param error:
    :return: rv with a machine error added on
    """
    assert len(rv) == len(error)
    for n in range(len(rv)):
        rv[n] +=  error[n]*np.random.normal(0, 1, len(error[n]))*u.km/u.s
    return rv
