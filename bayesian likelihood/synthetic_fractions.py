from synthetic_galaxy import synthetic_galaxy
from error_with_jitter import error_with_jitter
from machine_error import machine_error
from binary_fraction import binary_fraction


def synthetic_fractions(num_of_galaxies, cloud, bf, m_min, mu, sigma, a, b):
    """Makes a list of detection rates from synthetic simulated galaxies

    Inputs
    ----------
    num_of_galaxies:    How many synthetic galaxies you want to make
    cloud:              Astropy table object. The table is used as a basis for making the synthetic galaxies. Should have
                        been made through the DataSimp process.
    bf:                 Desired binary fraction that you want in the synthetic galaxy. Should be a number between [0,1]
    m_min:              Minimum mass you want to consider for a secondary object. Needs to have astropy units of .JupyterMass
    mu:                 Mean value used in making the period distribution for binary orbits
    sigma:              Standard deviation for the period distribution for the binary orbits
    a:                  Jitter parameter, y-intercept for the line in log space
    b:                  Jitter parameter, slope for the line in log space.

    Output
    ----------
    syn_fraction:   List of detection rates for each synthetic galaxy made. Has len = num_of_galaxies
    """
    # Counting variable
    galaxy_count = 0
    # Empty list used to store the final answers.
    syn_fractions = []
    while galaxy_count < num_of_galaxies:
        # Makes a list of synthetic radial velocity and error observations.
        rv_syn, err_syn = synthetic_galaxy(cloud, bf, m_min, mu, sigma)

        # Adds the extra jitter to the errors
        err_syn_j = error_with_jitter(err_syn, a, b, cloud['LOGG'])

        # Makes the rv values deviate from the exact values, using the error with jitter
        rv_syn_j = machine_error(rv_syn, err_syn_j)

        # Finds the detection rate for the synthetic radial velocity and error values.
        syn_fraction_j = binary_fraction(rv_syn_j, err_syn_j)

        # Append the list we return
        syn_fractions.append(syn_fraction_j)
        # Increase the counting variable by one
        galaxy_count += 1
    return syn_fractions