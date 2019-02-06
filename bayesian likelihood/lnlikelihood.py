import numpy as np
from synthetic_fractions import synthetic_fractions


def lnlikelihood(real_data_detection_rate, num_of_galaxies, cloud, bf, m_min, mu, sigma, a, b):

    # Array of detection rate for synthetic signals.
    syn_fractions = synthetic_fractions(num_of_galaxies, cloud, bf, m_min, mu, sigma, a, b)
    mean = np.mean(syn_fractions)
    std = np.std(syn_fractions)
    # print(mean, std, (real_data_detection_rate - mean)**2 / std**2, np.log(2*np.pi * std**2),
    #       ((real_data_detection_rate - mean) ** 2 / std ** 2 + np.log(2 * np.pi * std ** 2)),
    #       -1 / 2 * ((real_data_detection_rate - mean) ** 2 / std ** 2 + np.log(2 * np.pi * std ** 2)))
    lnl = -1/2*((real_data_detection_rate - mean)**2 / std**2 + np.log(2*np.pi * std**2))
    return lnl
