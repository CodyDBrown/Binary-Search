import numpy as np
from lnprior import lnprior
from lnlikelihood import lnlikelihood
from error_with_jitter import error_with_jitter
from binary_fraction import binary_fraction


def lnprob(theta, num_of_galaxies, cloud, m_min):
    bf, a, b, mu, sigma = theta

    error_j = error_with_jitter(cloud['RADIAL_ERR'],
                                                 a,
                                                 b,
                                                 cloud['LOGG'])

    real_data_detection_rate = binary_fraction(cloud['RADIALV'],
                                               error_j)
    lp = lnprior(bf, a, b, mu, sigma)
    if not np.isfinite(lp):
        return -np.inf
    print("Done with binary fraction, ", bf)
    return lp + lnlikelihood(real_data_detection_rate, num_of_galaxies, cloud, bf, m_min, mu, sigma, a, b)
