"""
Makes a galaxy of synthetic observations, based on a table of real observations
"""
from binary_params import binary_params
from rv_from_param import rv_from_param
import astropy.units as u
import numpy as np


def synthetic_galaxy(cloud, bf, m_min, mu, sigma):
    """Makes a list of radial velocity values and errors for a synthetic galaxy with the desired parameters. This program
    uses real observations as a basis for making the synthetic observations. That way

    Inputs
    ----------
    cloud:              Astropy table object. The table is used as a basis for making the synthetic galaxies. Should have
                        been made through the DataSimp process.
    bf:                 Desired binary fraction that you want in the synthetic galaxy. Should be a number between [0,1]
    m_min:              Minimum mass you want to consider for a secondary object. Needs to have astropy units of .JupyterMass
    mu:                 Mean value used in making the period distribution for binary orbits
    sigma:              Standard deviation for the period distribution for the binary orbits

    Output
    ----------
    rv_syn_galaxy:      List of synthetic radial velocity observations and observational errors.
    """
    # Empty array to put answers into
    rv_syn_galaxy = []
    # Make the units of the radial_err column km/s. This should be done in DataSimp, but that's a future task
    cloud['RADIAL_ERR'].unit = u.km/u.s

    # Get the real observed error values and use those for the synthetic erros
    error = cloud['RADIAL_ERR']
    for n in range(len(cloud)):
        # Deside if we are going to make a binary star or a solo star. When bf = 1, then we are always in a binary if
        # bf = 0 then the 'if' statement is always false and we always have a solo star.
        if bf > np.random.uniform(): # Then we observe a binary

            # Make a dictionary of orbital parameters that will be used to make
            buddy_dictionary = binary_params(m_min, mu, sigma,
                                             cloud['ISO_RAD'][n]*u.solRad,
                                             cloud['ISO_MASS'][n]*u.solMass)

            syn_binary_rv = rv_from_param(cloud['VHELIO_AVG'][n]*u.km/u.s, buddy_dictionary['k'],
                                          buddy_dictionary['p'].value, buddy_dictionary['e'],
                                          buddy_dictionary['phi'], buddy_dictionary['w'],
                                          date = cloud['RADIAL_DATE'][n])

            rv_syn_galaxy.append(syn_binary_rv)

        else: # Then we have a solo star and don't need to make synthetic cbinaries
            syn_solo_rv = np.array([cloud['VHELIO_AVG'][n]]*len(cloud['RADIALV'][n]))*u.km/u.s
            rv_syn_galaxy.append(syn_solo_rv)

    return rv_syn_galaxy, error








