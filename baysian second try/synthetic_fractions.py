from synthetic_galaxy import synthetic_galaxy
from error_with_jitter import error_with_jitter
from machine_error import machine_error
from binary_fraction import binary_fraction


def synthetic_fractions(num_of_galaxies, cloud, bf, m_min, mu, sigma, a, b):
    galaxy_count = 0
    syn_fractions = []
    while galaxy_count < num_of_galaxies:
        rv_syn, err_syn = synthetic_galaxy(cloud, bf, m_min, mu, sigma)
        # For some reason rv_syn doesn't have units.... so lets add them
        #rv_syn_units = rv_syn * u.km/y.s

        err_syn_j = error_with_jitter(err_syn, a, b, cloud['LOGG'])
        rv_syn_j = machine_error(rv_syn, err_syn_j)

        syn_fraction_j = binary_fraction(rv_syn_j, err_syn_j)
        syn_fractions.append(syn_fraction_j)
        print("Done with ", galaxy_count)
        galaxy_count += 1
    return syn_fractions