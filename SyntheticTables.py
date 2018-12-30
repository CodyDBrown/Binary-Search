"""
The goal here is to make a large sample of synthetic observations a head of time so that when I'm finding the beta values
I just need to pull simulations from this table rather than make new values. My worry is that I will need to make an
extremely large pool of simulations in order to make the results statisticly significent. The current way I have things
set up if I have 24 walkers, taking 25 steps, then for the LMC I'm making about 7 billion stars. That's a lot of time.
So to save on time if I can just make a list of say, 100,000 stars of both binaries and solo stars, then I can just pull
from that list rather than make a new one every time. But my worry is that I will have substantially fewer stars than
when I make new ones every time.
"""

import numpy as np
from scipy import optimize
import astropy.units as u
from astropy.constants import G, sigma_sb, c
from astropy.table import Table, Column, vstack


class SyntheticTables:

    def __init__(self, aas_table):
        self.aas_table = aas_table

    def buddy_values(self, N, m_min, period):

        # Stratified sampling making full tables for this.

        """
        Makes a set of orbital paramiters from random distributions, to be used with other programs to make our collection
        of modled observations

        Inputs
        ----------
        N:      Row number of the table we want to look at
        m_min:  Minimum companion mass we want to consider Should be in Jupiter masses
        period: List of period distribution paramiters we want to look at. First element should be a string either "L" or
                "U" for lognormal or Uniform. The other two elements will depend on what kind of distribution we want to
                look at, either
                ["L", mu, sigma] or
                ["U", min, max]
                Units of the period is in days

        """
        r_peri = 0 * u.solRad  # Initial set up to be zero so we enter the while loop
        in_case_of_emergency = 0  # Variable that will get us out of the loop if we're stuck forever
        while r_peri.value < 1.1 * self.aas_table['ISO_MEANR'][N]:
            M = self.aas_table['ISO_MEANM'][N] * u.solMass
            # Make the fake companion that we want orbiting our primary star
            m_buddy = np.random.uniform(m_min, M.to(
                u.jupiterMass).value) * u.jupiterMass  # For reference the 1 solMas = 1047 jupMas

            # Checks what kind of period distribution we want
            if period[0] == 'U':
                P_buddy = np.random.uniform(period[1], period[2]) * u.d
            elif period[0] == 'L':
                P_buddy = (10 ** np.random.normal(period[1], period[2])) * u.d
                # Upper bound on what the period can be. Just taken from Spencer
                # TODO Calculate this for the LMC and SMC. Right now i'm just using this from the Spencer papter but It
                # should be calculated for each galaxy
                while P_buddy.value > 10 ** 6:
                    P_buddy = (10 ** np.random.normal(period[1], period[2])) * u.d

            # Catch if input is wrong.
            else:
                return print('Period flag needs to be "L" or "U" not {}'.format(period))

            # Calculate the semi-major axis for the orbit
            a_buddy = np.cbrt(((G * (M + m_buddy)) / (4 * np.pi ** 2)) * P_buddy ** 2)
            a_buddy = a_buddy.to(u.AU)  # Convert it to AU
            if P_buddy.value < 12:
                e_buddy = 0 * u.one
            else:
                e_buddy = np.random.uniform(0, 0.93) * u.one
            n_foo = (2 * np.pi) / P_buddy

            # Make sure the closest point of the orbit isn't so close that we have to worry about title effects.
            r_peri = (1 - e_buddy) * a_buddy
            r_peri = r_peri.to(u.solRad)

            in_case_of_emergency += 1
            if in_case_of_emergency > 20:
                print("You got stuck!")  # #TODO Need to come up with a better way to handle these cases.
                break

        # Also need some angles of the orbit that we would see.
        i_buddy = np.random.uniform(0, np.pi) * u.rad
        # i_buddy = np.pi/2*u.rad
        # These are some phase angles that depend on when we first see it.
        w_buddy = np.random.uniform(0, np.pi) * u.rad
        phi_buddy = np.random.uniform(0, np.pi) * u.rad

        # Now I can find the "K" paramiter based on these values.
        K_buddy = (m_buddy / (M + m_buddy)) * (n_foo * a_buddy * np.sin(i_buddy)) / np.sqrt(1 - e_buddy ** 2)
        K_buddy = K_buddy.to(u.km / u.s)
        buddy_dict = {'m': m_buddy, 'e': e_buddy, 'P': P_buddy, "a": a_buddy, "i": i_buddy,
                      "w": w_buddy, "phi": phi_buddy, "K": K_buddy, "ID": self.aas_table["APOGEE_ID"][N]}

        return buddy_dict

    """ 
    Define a bunch of smaller functions I need to get the radial velocity values
    """

    def _mean_anom(self, Date, P, phi):
        return 2 * np.pi * Date / P - phi

    def _g(self, E, M, ec):
        return E - ec * np.sin(E) - M

    def _gp(self, E, M, ec):
        return 1 - ec * np.cos(E)

    """
    Use the scipy.optomize.newton for _NR_E and _NR_f
    """

    def _NR_E(self, M, ec):
        E = []
        for m in M:
            e_i = m  # Start with E_i = M
            delta = 1  # Give it a value to enter the loop
            while abs(delta) > 1e-8:
                e_i1 = e_i - self._g(e_i, m, ec) / self._gp(e_i, m, ec)
                delta = e_i1 - e_i
                e_i = e_i1
                # print(e_i, e_i1, delta)

            assert self._g(e_i1, m, ec) < 1e-8  # Double check things work the right way
            E.append(e_i1)
        print('Function check for my newton', self._g(E, M, ec))
        return np.array(E)

    def _h(self, f, E, ec):
        return np.cos(f) - (np.cos(E) - ec) / (1 - ec * np.cos(E))

    def _hp(self, f, E, ec):
        return -1 * np.sin(f)

    # Make the Jitter a function that we call
    def _jitter(self, N, a, b):
        """
        Modle the jitter using different paramiters using an equation 10^(a - b*logg). The default values give a curve
        really similar to the Hecker equation 2*0.015**(1/3*logg)
        :param N: Row of the star you're on
        :param a:
        :param b:
        :return: jit
        """
        return 10 ** (a - b * self.aas_table['LOGG'][N])

    def fake_rv_binary(self, N, m_min, period,):
        """
        After I have a set of paramiters form the buddy_values I want to make a fake Radial Velocity Measurment
        based on those values and the Primary Stars Values.
        """
        buddy_dict = self.buddy_values(N, m_min, period)
        date = self.aas_table['RADIAL_DATE'][N]


        err = self.aas_table['RADIAL_ERR'][N]

        M = 2 * np.pi / buddy_dict['P'].value * date - buddy_dict['phi'].value

        ec = buddy_dict['e'].value

        E0 = M.copy()
        E = optimize.newton(self._g, E0, fprime=self._gp, args=(M, ec),)

        assert all(self._g(E, M, ec) < 1e-7)
        f0 = E.copy()
        f = optimize.newton(self._h, f0, fprime=self._hp, args=(E, buddy_dict['e'].value))

        rv_buddy = buddy_dict['K'] * (np.cos(buddy_dict['w'].value + f) + buddy_dict['e'].value * np.cos(buddy_dict['w']))
        #print(rv_buddy.value)
        rv_buddy = self.aas_table['VHELIO_AVG'][N] + rv_buddy.value
        #print(rv_buddy)

        rv_buddy += err * np.random.normal(0, 1, len(rv_buddy))  # This should be the same as the for loop in the comment
        # print("Did we end?")
        return rv_buddy, err, date, buddy_dict


    def buddy_table(self, buddy_dict):
        """
        Takes the output from buddy_values and turns it into an astropy table. The goal is to have a table of
        values from the simulations. Some of
        """


        b_table = Table([[False],
                           [buddy_dict['m'].value], [buddy_dict['e'].value], [buddy_dict["P"].value], [buddy_dict["a"].value],
                           [buddy_dict["i"].value], [buddy_dict["w"].value], [buddy_dict['phi'].value], [buddy_dict["K"].value],
                           [0], [buddy_dict["ID"]]
                           ],
                          names = ('Binary','m','e','P','a','i','w','phi','K','P-value','APOGEE_ID'),
                          dtype=('b','f8','f8','f8','f8','f8','f8','f8','f8','f8','str')
                         )
        return b_table

    def buddy_array_maker(self, m_min, period, times):

        # Because I'm a good programmer I'm going to make empty arrays of zeros to store my answers in. This is to make
        # sure that we alocate enough memory when we start and don't run into memory problems an hour into the program running

        secondary_mass_array = np.zeros(len(self.aas_table) * times)
        eccentricity_array = np.zeros(len(self.aas_table) * times)
        period_array = np.zeros(len(self.aas_table) * times)
        semi_major_axis_array = np.zeros(len(self.aas_table) * times)
        inclination_angle_array = np.zeros(len(self.aas_table) * times)
        omega_array = np.zeros(len(self.aas_table) * times)
        phi_array = np.zeros(len(self.aas_table) * times)
        k_array = np.zeros(len(self.aas_table) * times)

        radial_velocity_array = [None]*(len(self.aas_table) * times)
        error_array = [None]*(len(self.aas_table) * times)
        date_array =  [None]*(len(self.aas_table) * times)

        m1_array =  np.zeros(len(self.aas_table) * times)
        logg_array = np.zeros(len(self.aas_table) * times)
        time = 0
        while time < times:
            for n in range(len(self.aas_table)):
                #print(len(self.aas_table)*time + n)
                radial_velocity, error, date, buddy_dictionary = self.fake_rv_binary(n, m_min, period)


                radial_velocity_array[len(self.aas_table)*time + n] = str(radial_velocity)
                error_array[len(self.aas_table)*time + n] = str(error)
                date_array[len(self.aas_table)*time + n] = str(date)

                m1_array[len(self.aas_table)*time + n] = self.aas_table['ISO_MEANM'][n]
                logg_array[len(self.aas_table)*time + n] = self.aas_table['LOGG'][n]

                secondary_mass_array[len(self.aas_table)*time + n] = buddy_dictionary['m'].value
                eccentricity_array[len(self.aas_table)*time + n] = buddy_dictionary['e']
                period_array[len(self.aas_table)*time + n] = buddy_dictionary['P'].value
                semi_major_axis_array[len(self.aas_table)*time + n] = buddy_dictionary['a'].value
                inclination_angle_array[len(self.aas_table)*time + n] = buddy_dictionary['i'].value
                omega_array[len(self.aas_table)*time + n] = buddy_dictionary['w'].value
                phi_array[len(self.aas_table)*time + n] = buddy_dictionary['phi'].value
                k_array[len(self.aas_table)*time + n] = buddy_dictionary['K'].value
            time += 1
        return radial_velocity_array, error_array, date_array, m1_array,secondary_mass_array, eccentricity_array, period_array, \
               semi_major_axis_array, inclination_angle_array, omega_array, phi_array, k_array, logg_array

    def table_maker(self,  m_min, period, times):
        arrays = self.buddy_array_maker( m_min, period, times)
        b_table = Table( [arrays[0], arrays[1], arrays[2],
                          arrays[3], arrays[4], arrays[5],
                          arrays[6], arrays[7], arrays[8],
                          arrays[9], arrays[10], arrays[11],
                          arrays[12]],
                         names=( 'str_rv', 'str_err', 'str_date', 'M1','m2', 'e', 'P', 'a','i', 'w', 'phi', 'K', "logg"),
                         dtype=('str', 'str', 'str', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8','f8', 'f8')
                         )
        return b_table

