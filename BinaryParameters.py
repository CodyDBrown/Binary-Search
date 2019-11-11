import numpy as np
import astropy.units as u
from astropy.constants import G, sigma_sb, c


def BinaryParameters(minimumSecondaryMass, periodMean, periodSTD, primaryStarRadius, primaryStarMass):
    """
    Find a set of parameters to use for making a binary orbit
    :param minimumSecondaryMass: Minimum mass I want to consider (Jupyter Masses)
    :param mu: Mean value I want to use for the log normal period
    :param sigma: Standard deviation (std) for the log normal period distribution
    :return:
    """

    orbitPericenter = 0 * u.solRad
    emergency = 0

    # Keep picking values until the distance to pericenter is large enough
    while orbitPericenter < 5 * primaryStarRadius:
        # Should this be mass fraction, rather than a set minimum mass value?
        secondaryMass = np.random.uniform(minimumSecondaryMass.value, primaryStarMass.to(u.jupiterMass).value) * u.jupiterMass

        orbitPeriod = 10 ** np.random.normal(periodMean, periodSTD) *u.d
        MAXPERIOD = 10**6
        while orbitPeriod.value > MAXPERIOD: # Upper bound on the period. Should look into this a bit more.
            orbitPeriod = 10 ** np.random.normal(periodMean, periodSTD) * u.d

        semiMajorAxis = np.cbrt(((G * (primaryStarMass + secondaryMass)) / (4 * np.pi ** 2)) * orbitPeriod**2)
        semiMajorAxis = semiMajorAxis.to(u.AU) # Convert to AU
        MinEcentricPeriod = 12 * u.d
        if orbitPeriod < MinEcentricPeriod:
            eccentricity = 0
        else:
            eccentricity = np.random.uniform(0, 0.93)

        orbitPericenter = ((1 - eccentricity) * semiMajorAxis).to(u.solRad)
        emergency += 1
        if emergency > 100:
            orbitPeriod = 10 ** 6 *u.d
            semiMajorAxis = np.cbrt(((G * (primaryStarMass + secondaryMass)) / (4 * np.pi ** 2)) * orbitPeriod ** 2)
            semiMajorAxis = semiMajorAxis.to(u.AU)  # Convert to AU
            eccentricity = 0
            orbitPericenter = ((1 - eccentricity) * semiMajorAxis).to(u.solRad)

            if orbitPericenter < 5*star_radius:
                print("You got stuck")
                break
    n_holder = (2*np.pi) / orbitPeriod

    # Need to make 3 angle variabels

    i_buddy, omega_buddy, phi_buddy = np.random.uniform(0, np.pi, 3)

    k_buddy = ((secondaryMass / (primaryStarMass + secondaryMass)) * (n_holder * semiMajorAxis * np.sin(i_buddy)) /\
              np.sqrt(1 - eccentricity ** 2)).to(u.km/u.s)

    buddy_dict = {'m': secondaryMass, 'e': eccentricity, 'p': orbitPeriod, "a": semiMajorAxis,
                  "i": i_buddy, "w": omega_buddy, "phi": phi_buddy, "k": k_buddy,}
    
    return buddy_dict
