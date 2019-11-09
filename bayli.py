"""
I am trying to turn all of my bayesian likelihood functions into one module that will hopefully be easier to run than
all the smaller functions
"""

import os
import numpy as np
import astropy.units as u
from astropy.constants import G, sigma_sb, c
from scipy.stats import chi2
from scipy import optimize
from astropy.table import Table
import datetime as dt
import pickle as rick
import time

import emcee
from multiprocessing import Pool


def binary_detection(rv, error):
    """
    Determines if the rv and error are in a binary or not.
    Inputs
    ----------
    rv:     Radial velocity


    error:  errors for the measurments
    :return: Boolian, true if the rv and error gives a reasonable binary, false if not
    """

    assert len(rv) == len(error)  # Sanity check to make sure I have the same number of data points for rv and error
    chi_squared = chi_sq_mean(rv, error)

    p_value = 1 - chi2.cdf(chi_squared, len(rv) - 1)

    return p_value < 0.05


def binary_fraction(rv_list, error_list):
    assert len(rv_list) == len(error_list)
    detection = 0
    for n in range(len(rv_list)):
        # print(rv_list[n], error_list[n])
        binary = binary_detection(rv_list[n], error_list[n])

        if binary:
            detection += 1
    detection_rate = detection/len(rv_list)
    return detection_rate


def binary_params(m_min, mu, sigma, star_radius, star_mass):
    """
    :param m_min:       minimum mass we want to consider. Needs to have mass units
    :param mu:          Mean value for the Gaussian distribution used for picking the period
    :param sigma:       STD for the Gaussian distribution used for picking the period
    :param star_radius: Radius of the primary star. Needs to have distance units
    :param star_mass:   Mass of the primary star. Needs to have units of mass.
    :return buddy_dict: Dictionary of the orbital parameters
                        m: mass of secondadry object
                        e: eccentricity
                        p: period
                        a: semi-major axis
                        i: inclination angle
                        w: accention node
                        phi: abses to perricenter?
                        k: can't remember the name of this variable
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

        r_peri = ((1 - eccentricity)*semi_major_axis).to(u.solRad)
        emergency += 1
        if emergency > 50:
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


def chi_sq_mean(rv, errors):
    """
    Finds the chi-squared for RV data using the mean value as the model we fit to.

    :param rv:      Numpy array of radial velocity values. Should have units of speed
    :param errors:  Numpy array of errors for each radial velocity value. Should have the same units as rv
    :return:        Returns the chi-squared of the mean value model to the data.
    """
    return np.sum((rv - np.mean(rv))**2 / errors**2)


def error_with_jitter(error, c1, c2, logg):
    """
    Takes the errors for each radial velocity measurement and adds an extra jitter term onto it.

    :param error:   Numpy array of errors for each radial velocity measurement
    :param c1:      Parameter for our jitter model. Gives the y intersept in a log plot
    :param c2:      Parameter for out jitter model. Gives the slope in a log plot
    :param logg:    Surface gravity of the observed star.
    :return:        Observed error added to the jitter error in quadriture.
    """
    jit = jitter(c1, c2, logg)

    return (error**2 + jit.value**2)**0.5  # For some reason np.sqrt() wasn't working


def jitter(c1, c2, logg, cutoff = 1.0):
    """
    Finds the theoretical 'jitter' that will be added to the noise
    :param cutoff:
    :param c1: y-intersept in log space, when a = 0.3 it matches the value in Hecker/Troup
    :param c2: slope of the jitter in log space, when b = 0.61 it matches the value in Hecker/Troup
    :param logg: Surface gravity
    :return: jitter
    """

    """  2019-07-04: Changing the way we do the jitter model. Using a formula that dave had of 
    v_jitter = (10**(0.15 - 1*logg)) < 0.650"""
    # upper_bound = 1.2
    foo = [(10**(c1 + c2*lg)) if (10**(c1 + c2*lg)) < cutoff else cutoff for lg in logg]*u.km/u.s

    # return 10 ** (c1 + c2 * logg) * u.km/u.s
    return foo


def likelihood_result(c1, gal_path, c2=-0.3, mu=3.4, sigma=2.4, m_min=100*u.jupiterMass):
    gal = rick.load(open(gal_path, "rb"))

    fractions = np.linspace(0, 1, 25)

    gal_err_j = error_with_jitter(gal['RADIAL_ERR'], c1, c2, gal['LOGG'])

    gal_detection_rate_j = binary_fraction(gal['RADIALV'], gal_err_j)

    lnl = []
    for n in range(len(fractions)):
        lnl.append(lnlikelihood(gal_detection_rate_j,
                                75,
                                gal,
                                fractions[n],
                                m_min,
                                mu,
                                sigma,
                                c1,
                                c2)
                   )
        print("Done with,", n, dt.datetime.now())
    return lnl


def lnlikelihood(real_data_detection_rate, num_of_galaxies, cloud, bf, m_min, mu, sigma, c1, c2):

    # Array of detection rate for synthetic signals.
    syn_fractions = synthetic_fractions(num_of_galaxies, cloud, bf, m_min, mu, sigma, c1, c2)

    mean = np.mean(syn_fractions)
    std = np.std(syn_fractions)
    lnl = -1/2*((real_data_detection_rate - mean)**2 / std**2 + np.log(2*np.pi * std**2))
    return lnl


def lnprior(bf, c1):
    if 0.01 < bf < 1 and 0.4 < c1 < 1.0:
        return 0
    return -np.inf


def lnprob(theta, num_of_galaxies, cloud, m_min, mu, sigma, c2):
    bf, c1 = theta

    error_j = error_with_jitter(cloud['RADIAL_ERR'],
                                c1,
                                c2,
                                cloud['LOGG'])

    real_data_detection_rate = binary_fraction(cloud['RADIALV'],
                                               error_j)
    lp = lnprior(bf, c1)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlikelihood(real_data_detection_rate, num_of_galaxies, cloud, bf, m_min, mu, sigma, c1, c2)


def machine_error(rv, error):
    """
    Adds an extra machine error to the rv values based on the size of the error. Used for making the
    synthetic rv signals

    :param rv:
    :param error:
    :return: rv with a machine error added on
    """
    assert len(rv) == len(error) # Double check make sure that there are the same number of rv observations as errors
    for n in range(len(rv)):
        rv[n] += error[n]*np.random.normal(0, 1, len(error[n]))*u.km/u.s
    return rv


""" 
Define a bunch of smaller functions I need to get the radial velocity values

_gp and _hp need the same arguments as _g and _h in order to work with newton
"""


def _mean_anom(date, p, phi):
    return 2 * np.pi * date / p - phi


def _g(e, m, ec):
    return e - ec * np.sin(e) - m


def _gp(e, m, ec):
    return 1 - ec * np.cos(e)


def _h(f, e, ec):
    return np.cos(f) - (np.cos(e) - ec) / (1 - ec * np.cos(e))


def _hp(f, e, ec):
    return -1 * np.sin(f)


def rv_from_param(v0, k, period, ec, phi, omega, date):
    """
    Finds the radial velocity along our line of sight from some of the orbital paramiters
    :param v0: Barrycenter velocity
    :param k: Don't remember the name of this, but it's like the velocity in it's refference frame
    :param period: period of the orbit
    :param ec: eccentricity of the orbit
    :param phi:
    :param omega:
    :param date: times I want to know the radial velocity at
    :return rv: Radial velocity in our line of sight for the input dates
    """
    M = _mean_anom(date, period, phi)

    E0 = M.copy()
    E = optimize.newton(_g, E0, fprime=_gp, args=(M, ec))

    f0 = E.copy()
    f = optimize.newton(_h, f0, fprime=_hp, args=(E, ec))

    in_frame_rv = np.cos(omega + f) + ec*np.cos(omega)
    rv = k * in_frame_rv
    rv += v0

    return rv


def same_data(table1, table2, temp_lim=250, logg_lim=0.2, feh_lim=0.2,):
    """
    Makes the data between two tables similar with in some limit

    Inputs
    ---------
    table1:  Should be the larger table that we want cut down
    table2:  Should be the smaller table that we want to compare the larger table to
    limit:   Tollerence that we want to compare the data to.

    Output
    ----------
    table1:  Modified table1 only keeping data points that are close to table2
    """
    # For each row in all_average_data, find isochrone rows that have similar
    # values, and make a list of all of those entries.
    dtype_list = []
    table1_foo = Table(table1, copy=True)
    table2_foo = Table(table2, copy=True)
    for n in range(len(table1_foo.dtype)):
        dtype_list.append(table1_foo.dtype[n])

    table3_foo = Table(names=table1_foo.colnames, dtype=dtype_list)
    rr = []
    for j in range(len(table2)):
        gd, = np.where((np.abs(table1_foo['TEFF'] - table2_foo['TEFF'][j]) < temp_lim) &
                       (np.abs(table1_foo['LOGG'] - table2_foo['LOGG'][j]) < logg_lim) &
                       (np.abs(table1_foo['FE_H'] - table2_foo['FE_H'][j]) < feh_lim)  # &
                       # (np.abs(table1_foo['VERR'] - table2['VERR'][j]) < snr_lim )
                       )

        # print(gd, type(gd))
        if len(gd) >= 1:
        #     table3_foo.add_row(table1_foo[gd[0]])
        #     table1_foo.remove_row(gd[0])
        #     # print(len(table1_foo))
        # elif len(gd) > 2:
            rand = np.random.randint(0, len(gd))
            table3_foo.add_row(table1_foo[gd[rand]])
            table1_foo.remove_row(gd[rand])
        else:
            rr.append(j)

    table2_foo.remove_rows(rr)

    return table3_foo, table2_foo


def synthetic_fractions(num_of_galaxies, cloud, bf, m_min, mu, sigma, a, b):
    """Makes a list of detection rates from synthetic simulated galaxies

    Inputs
    ----------
    num_of_galaxies:    How many synthetic galaxies you want to make
    cloud:              Astropy table object. The table is used as a basis for making the synthetic galaxies. Should
                        have been made through the DataSimp process.
    bf:                 Desired binary fraction that you want in the synthetic galaxy. Should be a number between [0,1]
    m_min:              Minimum mass you want to consider for a secondary object. Needs to have astropy units
                        of .JupyterMass
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


def mcmc_likelihood(gal, ndim, nwalkers, nsteps, outfile, gal_per_run = 75,
                    mu=None, sigma=None,  c2 = None):

    gal['RADIALV'].unit = u.km/u.s
    gal['RADIAL_ERR'].unit = u.km/u.s

    start_val = [np.array([0.5, 0.7]) + 0.2*np.random.randn(ndim) for i in range(nwalkers)]
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                        pool = pool,
                                        args=(gal_per_run,
                                              gal,
                                              150*u.jupiterMass,
                                              mu, sigma, c2
                                              )
                                        )
                                        
        start = time.time()
        sampler.run_mcmc(start_val, nsteps, progress=True)
        end = time.time()
        serial_time = end - start
        print("Serial took {0:.1f} seconds".format(serial_time))
        #sampler.run_mcmc(start_val, nsteps, progress=True)
    del sampler.pool
    if os.path.exists(outfile):
        os.remove(outfile)
    rick.dump(sampler, open(outfile, "wb"))
