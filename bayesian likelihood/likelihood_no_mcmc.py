import numpy as np
from lnlikelihood import lnlikelihood
from error_with_jitter import error_with_jitter
from binary_fraction import binary_fraction
import astropy.units as u
import pickle


def lnlh(a, gal=None, path=None):
    if gal is None:
        gal = pickle.load(open(path, "rb"))
    if path is None and gal is None:
        print("gal or path need to be defined")

    b = -0.2
    mu = 3.2
    sigma = 2.4

    fractions = np.linspace(0, 1, 25)

    gal_err_j = error_with_jitter(gal['RADIAL_ERR'], a, b, gal['LOGG'])

    detection_rate_j = binary_fraction(gal['RADIALV'], gal_err_j)

    lnl = []
    for n in range(len(fractions)):
        lnl.append(lnlikelihood(detection_rate_j,
                                200,
                                gal,
                                fractions[n],
                                150 * u.jupiterMass,
                                mu,
                                sigma,
                                a,
                                b)
                   )
        print("Done with,", n)
    return lnl
