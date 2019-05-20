import numpy as np
import astropy.units as u
from astropy.constants import G, sigma_sb, c


def binary_params(m_min, mu, sigma, star_radius, star_mass):
    """
    Find a set of parameters to use for making a binary orbit

    :param m_min: Minimum mass I want to consider (Jupyter Masses)
    :param mu: Mean value I want to use for the log normal period
    :param sigma: Standard deviation (std) for the log normal period distribution
    :return:
    """

    r_peri = 0 * u.solRad
    emergency = 0

    # Keep picking values until the distance to pericenter is large enough
    while r_peri < 5 * star_radius:
        # Should this be mass fraction, rather than a set minimum mass value?
        m_buddy = np.random.uniform(m_min.value, star_mass.to(u.jupiterMass).value) * u.jupiterMass

        p_buddy = 10 ** np.random.normal(mu, sigma) *u.d
        while p_buddy.value > 10**6: # Upper bound on the period. Should look into this a bit more.
            p_buddy = 10 ** np.random.normal(mu, sigma) * u.d

        semi_major_axis = np.cbrt(((G * (star_mass + m_buddy)) / (4 * np.pi ** 2)) * p_buddy**2)
        semi_major_axis = semi_major_axis.to(u.AU) # Convert to AU

        if p_buddy < 12*u.d:
            eccentricity = 0
        else:
            eccentricity = np.random.uniform(0, 0.93)

        r_peri = ((1 - eccentricity) * semi_major_axis).to(u.solRad)
        emergency += 1
        if emergency > 100:
            p_buddy = 10 ** 6 *u.d
            semi_major_axis = np.cbrt(((G * (star_mass + m_buddy)) / (4 * np.pi ** 2)) * p_buddy ** 2)
            semi_major_axis = semi_major_axis.to(u.AU)  # Convert to AU
            eccentricity = 0
            r_peri = ((1 - eccentricity) * semi_major_axis).to(u.solRad)

            if r_peri < 5*star_radius:
                print("You got stuck")
                break
    n_holder = (2*np.pi) / p_buddy

    # Need to make 3 angle variabels

    i_buddy, omega_buddy, phi_buddy = np.random.uniform(0, np.pi, 3)

    k_buddy = ((m_buddy / (star_mass + m_buddy)) * (n_holder * semi_major_axis * np.sin(i_buddy)) /\
              np.sqrt(1 - eccentricity ** 2)).to(u.km/u.s)

    buddy_dict = {'m': m_buddy, 'e': eccentricity, 'p': p_buddy, "a": semi_major_axis,
                  "i": i_buddy, "w": omega_buddy, "phi": phi_buddy, "k": k_buddy,}

    return buddy_dict



