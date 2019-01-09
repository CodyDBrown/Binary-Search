"""
Makes a galaxy of synthetic observations, based on a table of real observations
"""
from binary_params import binary_params
from rv_from_param import rv_from_param
import astropy.units as u
import numpy as np


def synthetic_galaxy(aas_table, bf, m_min, mu, sigma):
    rv_syn_galaxy = []
    aas_table['RADIAL_ERR'].unit = u.km/u.s
    error = aas_table['RADIAL_ERR']
    for n in range(len(aas_table)):
        if bf > np.random.uniform(): # Then we observe a binary

            buddy_dictionary = binary_params(m_min, mu, sigma,
                                             aas_table['ISO_MEANR'][n]*u.solRad,
                                             aas_table['ISO_MEANM'][n]*u.solMass)

            syn_binary_rv = rv_from_param(aas_table['VHELIO_AVG'][n]*u.km/u.s, buddy_dictionary['k'],
                                          buddy_dictionary['p'].value, buddy_dictionary['e'],
                                          buddy_dictionary['phi'], buddy_dictionary['w'],
                                          date = aas_table['RADIAL_DATE'][n])

            rv_syn_galaxy.append(syn_binary_rv)

        else: # Then we have a solo star and don't need to make synthetic cbinaries
            syn_solo_rv = np.array([aas_table['VHELIO_AVG'][n]]*len(aas_table['RADIALV'][n]))*u.km/u.s
            rv_syn_galaxy.append(syn_solo_rv)

    return rv_syn_galaxy, error








