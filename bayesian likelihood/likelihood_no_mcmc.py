import numpy as np
from lnlikelihood import lnlikelihood
from error_with_jitter import error_with_jitter
from binary_fraction import binary_fraction
import astropy.units as u
import pickle

def lnlh(a):
    b = 0.61
    gal = pickle.load(open("/home/cody/Binary Search/all-average-smc.pck", "rb" ) )
    mu = 3.2
    sigma = 2.4


    fractions = np.linspace(0, 1, 50)


    gal_err_j = error_with_jitter(gal['RADIAL_ERR'], a, b, gal['LOGG'])

    lmc_detection_rate_j = binary_fraction(gal['RADIALV'], gal_err_j)

    lnl = []
    for n in range(len(fractions)):
        lnl.append(lnlikelihood(lmc_detection_rate_j,
                                50,
                                gal,
                                fractions[n],
                                100 * u.jupiterMass,
                                mu,
                                sigma,
                                a,
                                b)
                   )
        print("Done with,", n)
    return lnl