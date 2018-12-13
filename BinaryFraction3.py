from astropy.table import Table, Column, vstack
import numpy as np
from scipy.stats import chi2
import astropy.units as u
from astropy.constants import G, sigma_sb, c

from astropy.time import Time

from thejoker.data import RVData
from twobody.wrap import cy_rv_from_elements
from twobody.utils import ArrayProcessor
from twobody import KeplerOrbit

class BinaryFraction3:
    def __init__(self,AAS_TABLE):
        self.AAS_TABLE = AAS_TABLE

    def buddy_values(self, N, m_min, period):
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
        r_peri = 0*u.solRad # Initial set up to be zero so we enter the while loop
        in_case_of_emergency = 0  # Variable that will get us out of the loop if we're stuck forever
        while r_peri.value < 1.1*self.AAS_TABLE['ISO_MEANR'][N]:
            M = self.AAS_TABLE['ISO_MEANM'][N] * u.solMass
            # Make the fake companion that we want orbiting our primary star
            m_buddy = np.random.uniform(m_min, M.to(u.jupiterMass).value) * u.jupiterMass  # For reference the 1 solMas = 1047 jupMas

            #Checks what kind of period distribution we want
            if period[0] == 'U':
                P_buddy = np.random.uniform(period[1],period[2])*u.d
            elif period[0] == 'L':
                P_buddy = (10**np.random.normal(period[1], period[2]))*u.d
                # Upper bound on what the period can be. Just taken from Spencer
                #TODO Calculate this for the LMC and SMC. Right now i'm just using this from the Spencer papter but It
                # should be calculated for each galaxy
                while P_buddy.value > 10**6.51:
                    P_buddy = (10 ** np.random.normal(period[1], period[2])) * u.d

            # Catch if input is wrong.
            else:
                return print('Period flag needs to be "L" or "U" not {}'.format(period))

            # Calculate the semi-major axis for the orbit
            a_buddy = np.cbrt(((G * (M + m_buddy)) / (4 * np.pi ** 2)) * P_buddy**2)
            a_buddy = a_buddy.to(u.AU) # Convert it to AU
            if P_buddy.value < 12:
                e_buddy = 0*u.one
            else:
                e_buddy = np.random.uniform(0, 0.93)*u.one
            n_foo = (2 * np.pi) / P_buddy

            # Make sure the closest point of the orbit isn't so close that we have to worry about title effects.
            r_peri = (1 - e_buddy) * a_buddy
            r_peri = r_peri.to(u.solRad)

            in_case_of_emergency += 1
            if in_case_of_emergency > 20:
                print("You got stuck!")  # #TODO Need to come up with a better way to handle these cases.
                break

        # Also need some angles of the orbit that we would see.
        i_buddy = np.random.uniform(0, np.pi)*u.rad

        #These are some phase angles that depend on when we first see it.
        w_buddy = np.random.uniform(0, np.pi)*u.rad
        phi_buddy = np.random.uniform(0, np.pi)*u.rad


        # Now I can find the "K" paramiter based on these values.
        K_buddy = (m_buddy / (M + m_buddy)) * (n_foo * a_buddy * np.sin(i_buddy)) / np.sqrt(1 - e_buddy ** 2)
        K_buddy = K_buddy.to(u.km / u.s)
        buddy_dict = {'m': m_buddy, 'e': e_buddy, 'P': P_buddy, "a": a_buddy, "i": i_buddy,
                      "w": w_buddy, "phi": phi_buddy, "K": K_buddy, "ID": self.AAS_TABLE["APOGEE_ID"][N]}
        #buddy_dict = [m_buddy, e_buddy, P_buddy, a_buddy, i_buddy, w_buddy,
        #           phi_buddy, K_buddy, self.AAS_TABLE['APOGEE_ID'][N]]
        return buddy_dict

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

    """ 
    Define a bunch of smaller functions I need to get the radial velocity values
    """
    def _mean_anom(self,Date, P, phi):
        return 2*np.pi*Date/P - phi

    def _g(self, E, M, ec):
        return E - ec*np.sin(E) - M

    def _gp(self, E, ec):
        return 1 - ec*np.cos(E)

    def _NR_E(self, M, ec):
        E = []
        for m in M:
            e_i = m  # Start with E_i = M
            delta = 1  # Give it a value to enter the loop
            while abs(delta) > 1e-8:
                e_i1 = e_i - self._g(e_i, m, ec) / self._gp(e_i, ec)
                delta = e_i1 - e_i
                e_i = e_i1
                # print(e_i, e_i1, delta)
            assert self._g(e_i1, m, ec) < 1e-8  # Double check things work the right way
            E.append(e_i1)
        return np.array(E)

    def _h(self, f, E, ec):
        return np.cos(f) - (np.cos(E) - ec) / (1 - ec * np.cos(E))

    def _hp(self, f):
        return -1 * np.sin(f)

    def _NR_f(self, E, ec):
        f = []
        for e in E:
            f_i = e  # Start with E_i = M
            delta = 1  # Give it a value to enter the loop
            while abs(delta) > 1e-8:
                f_i1 = f_i - self._h(f_i, e, ec) / self._hp(f_i)
                delta = f_i1 - f_i
                f_i = f_i1

            assert self._h(f_i1, e, ec) < 1e-8
            f.append(f_i1)
        return np.array(f)

    # Make the Jitter a function that we call
    def _jitter(self, N, a, b):
        """
        Modle the jitter using different paramiters using an equation 10^(a - b*logg). The default values give a curve
        really similar to the hecker equation 2*0.015**(1/3*logg)
        :param N: Row of the star you're on
        :param a:
        :param b:
        :return: jit
        """
        return 10 ** (a - b*self.AAS_TABLE['LOGG'][N])

    def fake_rv_binary(self, N, m_min, period, jitter, a, b):
        """
        After I have a set of paramiters form the buddy_values I want to make a fake Radial Velocity Measurment
        based on those values and the Primary Stars Values.
        """
        buddy_dict = self.buddy_values(N,m_min, period)
        date = self.AAS_TABLE['RADIAL_DATE'][N]

        if jitter:
            jit = self._jitter(N, a, b)

            err = (np.sqrt(self.AAS_TABLE['RADIAL_ERR'][N]**2 + jit**2) )*u.km/u.s
        else:
            err = self.AAS_TABLE['RADIAL_ERR'][N]*u.km/u.s

        # # This is some stuff that cy_rv_from_elements needs for it's time input. Not sure why but it doens't
        # # work without these few lines. i.e. DO NOT TOUCH!
        # t_buddy = Time(date, format = 'mjd')
        #
        # t_buddy = t_buddy.tcb.mjd
        #
        # procb = ArrayProcessor(t_buddy)
        #
        # t_buddy, = procb.prepare_arrays()
        #
        # td0 = t_buddy[0]
        #
        # td0 = Time(td0, format = 'mjd')
        #
        # # Makes the observed radial velocity in the binaries Barrycenter.
        # # buddy_dict = [m_buddy, e_buddy, P_buddy, a_buddy, i_buddy, w_buddy,
        # #           phi_buddy, K_buddy, self.AAS_TABLE['APOGEE_ID'][N]]
        #
        # rv_buddy = cy_rv_from_elements(t_buddy, buddy_dict["P"].to(u.day).value, 1. , buddy_dict["e"].value, buddy_dict["w"].value,
        #                                buddy_dict["phi"].value, td0.tcb.mjd,
        #                                anomaly_tol = 1E-10, anomaly_maxiter = 128)
        #
        # # Then we move the velocity to be in our reference frame. This is the exact analytical radial velcity of a star
        # # with the given paramiters.
        # rv_buddy = (buddy_dict["K"] * rv_buddy + self.AAS_TABLE['VHELIO_AVG'][N] * u.km/ u.s)

        M = 2 * np.pi / buddy_dict['P'].value * date - buddy_dict['phi'].value

        E = self._NR_E(M, buddy_dict['e'].value)
        f = self._NR_f(E, buddy_dict['e'].value)
        rv_buddy = buddy_dict['K']*(np.cos(buddy_dict['w'].value + f) + buddy_dict['e'].value * np.cos(buddy_dict['w']))
        rv_buddy += self.AAS_TABLE['VHELIO_AVG'][N]*u.km/u.s

        # Add on some machine error
        # for n in range(len(self.AAS_TABLE['RADIAL_ERR'][N])):
        #     rv_buddy[n] += np.random.normal(0, self.AAS_TABLE['RADIAL_ERR'][N][n]) * u.km/u.s
        for n in range(len(err)):
            rv_buddy[n] += np.random.normal(0, err[n].value) * u.km/u.s

        return rv_buddy, err, buddy_dict
    # TODO: Get rid of the reduce option. I never use it and it just causes more
    #       problems then it's worth.

    def chi_sq_mean(self, RV, err):
        """
        I want to fit some radial velocity measurments to a strait line of the average value. The logic goes
        that if the strait line fits the data well, then it's a single star floating in space. If a strait line
        fits the data poorly then there should be something there causing the deviation from the mean.

        Inputs
        ----------
        RV: Radial velocity values
        err: the error in those radial velocity values
        """
        chi_sq_value = np.sum((RV - np.mean(RV))**2 / err**2)
        return chi_sq_value

    def fake_binary_detection(self, N, m_min, period, jitter, a, b):
        """
        Uses the chi_sq_value to then find the P-Value then set the variable Binary in buddy table to be True
        or False depending on the threshold we set.
        """
        # First call the fake_rv function
        f_radial_velocity, f_err, bud_dict = self.fake_rv_binary(N, m_min, period, jitter, a, b)
        # Find the chi^2 for the fake rv values
        chi_squared = self.chi_sq_mean(f_radial_velocity, f_err,)
        # Find the p-value from that chi^2
        p_value = 1 - chi2.cdf(chi_squared, len(f_radial_velocity) - 1)
        # print(chi_squared, p_value)
        # Make the buddy table
        bud_table = self.buddy_table(bud_dict)
        # Set the 'P-value' in the buddy table
        bud_table['P-value'] = p_value
        # Check if it's a binary or not.
        if p_value < 0.05:
            f_binary = True
        else:
            f_binary = False
        # Put the result in the table
        bud_table['Binary'] = f_binary
        return f_radial_velocity, f_err, bud_table

    """
    Need the same thing but this time with no buddy, assumes a single star with error and decided if it's in a binary or not
    """
    def fake_solo_detection(self, N, jitter, a, b):
        if jitter:
            jit = self._jitter(N, a, b)
            solo_err = np.sqrt(self.AAS_TABLE['RADIAL_ERR'][N] ** 2 + jit ** 2)
        else:
            solo_err = self.AAS_TABLE['RADIAL_ERR'][N]
        # Make some fake solo RV measurments
        solo_RV = []
        for n in solo_err:
            rv_foo = np.random.normal(self.AAS_TABLE['VHELIO_AVG'][N], n)
            solo_RV.append(rv_foo)
        # Finds the chi_squared and p-value
        solo_chi_squared = self.chi_sq_mean(solo_RV, solo_err)
        p_value = 1 - chi2.cdf(solo_chi_squared, len(solo_RV) - 1)
        # Check if it's a binary or not.
        if p_value < 0.05:
            solo_binary = True
        else:
            solo_binary = False
        return p_value, solo_binary

    """
    Time to put it all together
    """

    def binary_or_not(self, N, m_min, period, b_fraction, jitter, a, b,):
        """
        Picks a random number and if it's below b_fraction then we run fake_binary_detection
        if it's larger than b_fraction then we run fake_solo_detection. Returns the P-value and
        the boolian 'Binary' for each run

        """
        # Pick a random number between 0 and 1.
        foo_random_number = np.random.uniform()
        if foo_random_number < b_fraction:
            foo_rv, foo_err, foo_table = self.fake_binary_detection(N, m_min, period, jitter, a, b)
            return foo_table['P-value'], foo_table['Binary']
        else:
            foo_pvalue, foo_binary = self.fake_solo_detection(N, jitter, a, b)
            return foo_pvalue, foo_binary


    def Master(self, m_min, period, b_fraction, loop, jitter, a, b):
        """
        Should just have to run this to get the detection rate for different binary
        fractions at the end of the day.
        """
        foo_exit = 0
        pvalue_list = []
        binary_bool_list = []
        while foo_exit < loop:
            for N in range(len(self.AAS_TABLE)):
                p_value, bool_binary = self.binary_or_not(N, m_min, period, b_fraction, jitter, a, b)
                pvalue_list.append(p_value)
                binary_bool_list.append(bool_binary)
            foo_exit += 1
        return pvalue_list, binary_bool_list

    def synthetic_detection_rate(self, m_min, period, b_fraction, loop, jitter, a=0.3, b=0.61):
        """
        Finds the synthetic detection rate for binary systems with the given input
        set of paramiters.

        Inputs
        ----------
        m_min:  Minimum companion mass I want to consider
        period: Either 'L' or 'U' for the lognormal period distribution or the
                uniform period distribution
                # TODO: I need to mess around with this more. Just found out that
                this using ln normal, not log10 normal, so I might need to fix?
        jitter: Boolian, if I want to add on jitter as an extra noise source,
                almost always set to True
        reduce: Boolian, if I want to use the reduced chi^2 or not. Should always
                be set to False becasue I'm looking for P-value not chi^2 value
                # TODO: GET RID OF THIS! It's always False so why have it around
        b_fraction: Binary fractions I want to consider. This should be an array
                    I've just been doing [0,1] but it could be a list of any value
        loop:   Number of times I want to loop through the table. Normaly set this
                to be between 10 and 50. More loops the more time this will take.

        Outputs
        ----------
        p_value_result: Array of the P-values for each of the different synthetic
                        binary
        binary_result_b:    Boolian array of if the P-Value was low enough to
                        determine if it was in a binary or not. Currently set to
                        0.05 but this can be changed in the 'fake_binary_detection'
                        and 'fake_solo_detection' programs
        detection_rate: Detection rate is kind of the answer I want. Array of
                        length equal to the length of the input b_fraction. Finds
                        the detection fraction based off the binary_result_foo for
                        each desired binary fraction
        detection_rate_error:   Finds the error in the detection rate. This just
                        assumes a shot noise as the source of error, so +/- sqrt(N)
                        this is a two element array for each element in detection_rate
                        first element will be the upper bound error, second will
                        be the lower bound error.
        """
        # Make empty arrays for the answers to be put in.
        detection_rate = []
        p_value_result = []
        binary_result_b = []
        detection_rate_error = []
        for i in b_fraction:
            p_value_foo, binary_result_foo = self.Master(m_min, period, i, loop, jitter, a, b)

            p_value_result.append(p_value_foo)
            binary_result_b.append(binary_result_foo)
            detection_rate.append(np.count_nonzero(binary_result_foo)/len(binary_result_foo))

            error = np.sqrt(np.count_nonzero(binary_result_foo))
            det_upper = (np.count_nonzero(binary_result_foo) + error) /len(binary_result_foo)
            det_lower = (np.count_nonzero(binary_result_foo) - error) /len(binary_result_foo)

            err_foo = [det_upper, det_lower]
            detection_rate_error.append(err_foo)
        return p_value_result, binary_result_b, detection_rate, detection_rate_error

    def Synthetic_Table(self, m_min, period, loop, jitter, a, b):
        """
        I want to make a table of all of the different synthetic paramiters I used
        And keep track of when the Binary detection gave True or False. That way
        I can look at the different cases where detections are made or not

        Inputs
        ----------
        m_min:  Minimum companion mass I want to consider
        period: Either 'L' or 'U' for the lognormal period distribution or the
                uniform period distribution
                # TODO: I need to mess around with this more. Just found out that
                this using ln normal, not log10 normal, so I might need to fix?
        jitter: Boolian, if I want to add on jitter as an extra noise source,
                almost always set to True
        reduce: Boolian, if I want to use the reduced chi^2 or not. Should always
                be set to False becasue I'm looking for P-value not chi^2 value
                # TODO: GET RID OF THIS! It's always False so why have it around
        loop:   Number of times I want to loop through the table. Normaly set this
                to be between 10 and 50. More loops the more time this will take.



        """
        foo_exit = 0
        foo_rv, foo_err, final_table = self.fake_binary_detection(0, m_min, period, jitter, a, b)
        while foo_exit < loop:
            for N in range(len(self.AAS_TABLE)):
                foo_rv, foo_err, foo_table = self.fake_binary_detection(N,m_min, period, jitter, a, b)
                final_table = vstack([final_table, foo_table])
            foo_exit += 1
        return final_table

    #-------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------

    """
    Same thing as above but now I want to be able to do it with the baysian method
    """

    def _beta(self, rv, err):
        """
        Finds the beta value for 1 star using equation 9 in the specer paper.

        Inputs
        ----------
        rv:     List of Radial velocity values
        err:    List of errors for each radial velocity

        Output
        ----------
        beta:   List of the different beta values for each combination of velocity points
        """
        assert len(rv) == len(err)  # Check to make sure they are the same length
        beta = []
        for n in range(len(rv)):
            for m in range(n + 1, len(rv)):
                v = np.abs(rv[m] - rv[n])
                errs = np.sqrt(err[n] ** 2 + err[m] ** 2)
                beta.append(v / errs)
        assert len(beta) == len(rv)*(len(rv) - 1)/2  # Again sanity check to make sure it worked
        return beta

    # Need to Find a beta distribution for Model data.
    def _betamod(self, N, m_min, period, b_fraction, jitter, a, b):
        binary_rng = np.random.uniform()
        if binary_rng < b_fraction:  # Then we're in a binary
            fake_rv, fake_err, buddy_dict = self.fake_rv_binary(N, m_min, period, jitter, a, b)
            beta = self._beta(fake_rv, fake_err)
            return beta
        else: # Solo star
            if jitter:
                jit = self._jitter(N, a, b)
                solo_err = np.sqrt(self.AAS_TABLE['RADIAL_ERR'][N] ** 2 + jit ** 2)
            else:
                solo_err = self.AAS_TABLE['RADIAL_ERR'][N]
            # Make some fake solo RV measurments
            solo_RV = []
            # for n in self.AAS_TABLE['RADIAL_ERR'][N]:
            #     rv_foo = np.random.normal(self.AAS_TABLE['VHELIO_AVG'][N], n)
            #     solo_RV.append(rv_foo)
            for n in solo_err:
                rv_foo = np.random.normal(self.AAS_TABLE['VHELIO_AVG'][N], n)
                solo_RV.append(rv_foo)
            beta = self._beta(solo_RV, solo_err)
            return beta
    def BetaMaster(self, m_min, period, b_fraction, loop, jitter, a = 0.3, b=0.61):
        foo_exit = 0
        beta_list = []
        while foo_exit < loop:
            for N in range(len(self.AAS_TABLE)):
                one_star_beta = self._betamod(N, m_min, period, b_fraction, jitter, a, b)
                beta_list.append(one_star_beta)
            foo_exit += 1
        flat_beta = [item for sublist in beta_list for item in sublist]
        return flat_beta

    def Real_Data_Beta(self,a = 0.3, b = 0.61):
        """
        Finds all of the beta values for the real data.
        :return: beta_data
        """
        beta_data = []
        for J in range(len(self.AAS_TABLE)):
            jit = self._jitter(J, a, b)
            #jit = 0
            err = np.sqrt(self.AAS_TABLE['RADIAL_ERR'][J]**2 + jit**2)
            beta = self._beta(self.AAS_TABLE['RADIALV'][J], err)
            beta_data.append(beta)
        beta_data = [item for sublist in beta_data for item in sublist]
        return beta_data
    """Does the binary fraction check for the real data table that was put into
    the Binary_Fraction when first initializing. I would also like to change the
    values

    """
    def Real_Data_Fraction(self, a = 0.3, b = 0.61):
        foo_check = []
        rd_p_value_array = []
        rd_Binary_array = []
        detection_rate_error = []
        for K in range(len(self.AAS_TABLE)):
            rd_rv = self.AAS_TABLE['RADIALV'][K]
            rd_err = self.AAS_TABLE['RADIAL_ERR'][K]
            rd_jitter = self._jitter(K, a, b)
            rd_err = np.sqrt(rd_err**2 + rd_jitter**2)
            rd_chi_squared = self.chi_sq_mean(rd_rv, rd_err,)
            rd_p_value = 1 - chi2.cdf(rd_chi_squared, len(rd_rv)-1)
            rd_p_value_array.append(rd_p_value)
        for i in rd_p_value_array:
            if i < 0.05:
                rd_Binary_array.append(True)
            else:
                rd_Binary_array.append(False)
        binary_detection = np.count_nonzero(rd_Binary_array)/len(rd_Binary_array)
        error = np.sqrt(np.count_nonzero(rd_Binary_array))
        det_upper = (np.count_nonzero(rd_Binary_array) + error) /len(rd_Binary_array)
        det_lower = (np.count_nonzero(rd_Binary_array) - error) /len(rd_Binary_array)
        err_foo = [det_upper, det_lower]
        detection_rate_error.append(err_foo)

        return rd_p_value_array, rd_Binary_array, binary_detection, detection_rate_error
