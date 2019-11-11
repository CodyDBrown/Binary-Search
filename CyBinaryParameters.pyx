import numpy as np
import astropy.units as u
import astropy.constants as con


def BinaryParameters(minimumSecondaryMass, float periodMean, float periodSTD, primaryStarRadius, primaryStarMass):
    # First thing I want to do is a unit strip. Make sure that the inputs are unitless and in mks.
    if (type(minimumSecondaryMass) == u.quantity.Quantity):
        minimumSecondaryMass = minimumSecondaryMass.to(u.kg) # turn it to kg
        minimumSecondaryMass = minimumSecondaryMass.value    # make it unitless
    if (type(primaryStarRadius) == u.quantity.Quantity):
        primaryStarRadius = primaryStarRadius.to(u.m) # turn it to m
        primaryStarRadius = primaryStarRadius.value    # make it unitless
    if (type(primaryStarMass) == u.quantity.Quantity):
        primaryStarMass = primaryStarMass.to(u.kg) # turn it to kg
        primaryStarMass = primaryStarMass.value    # make it unitless
    
    # Set of physics constants
    cdef double G = con.G.value
    
    # Set of all of my variables I will make in the file.     
    cdef double orbitPericenter = 0.0 
    cdef int emergency = 0
    cdef long int MAXPERIOD = 8.64*10**10  # 10**6 days in seconds. 
    cdef int SECONDS_IN_DAY = 86400
    cdef double orbitPeriod = 0.0
    cdef double semiMajorAxis = 0.0
    cdef double secondaryMass = 0.0
    cdef double i_buddy = 0
    cdef double omega_buddy = 0
    cdef double phi_buddy = 0


    # Keep picking values until the distance to pericenter is large enough
    while orbitPericenter < 5 * primaryStarRadius:
        # Should this be mass fraction, rather than a set minimum mass value?
        secondaryMass = np.random.uniform(minimumSecondaryMass, primaryStarMass)

        orbitPeriod = 10 ** np.random.normal(periodMean, periodSTD)*SECONDS_IN_DAY
        
        while orbitPeriod > MAXPERIOD: # Upper bound on the period. Should look into this a bit more.
            orbitPeriod = 10 ** np.random.normal(periodMean, periodSTD)*SECONDS_IN_DAY

        semiMajorAxis = np.cbrt(((G * (primaryStarMass + secondaryMass)) / (4 * np.pi ** 2)) * orbitPeriod**2)
        MinEcentricPeriod = 12 * SECONDS_IN_DAY
        if orbitPeriod < MinEcentricPeriod:
            eccentricity = 0
        else:
            eccentricity = np.random.uniform(0, 0.93)

        orbitPericenter = ((1 - eccentricity) * semiMajorAxis)
        emergency += 1
        if emergency > 100:
            orbitPeriod = MAXPERIOD
            semiMajorAxis = np.cbrt(((G * (primaryStarMass + secondaryMass)) / (4 * np.pi ** 2)) * orbitPeriod ** 2)
            
            eccentricity = 0
            orbitPericenter = ((1 - eccentricity) * semiMajorAxis)

            if orbitPericenter < 5*primaryStarRadius:
                print("You got stuck")
                break
    n_holder = (2*np.pi) / orbitPeriod

    # Need to make 3 angle variabels

    i_buddy, omega_buddy, phi_buddy = np.random.uniform(0, np.pi, 3)

    k_buddy = ((secondaryMass / (primaryStarMass + secondaryMass)) * (n_holder * semiMajorAxis * np.sin(i_buddy)) /\
              np.sqrt(1 - eccentricity ** 2))

    buddy_dict = {'m': secondaryMass, 'e': eccentricity, 'p': orbitPeriod, "a": semiMajorAxis,
                  "i": i_buddy, "w": omega_buddy, "phi": phi_buddy, "k": k_buddy,}
    
    return buddy_dict
