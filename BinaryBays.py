from astropy.table import Table, Column, vstack
import numpy as np
import scipy.stats as sps
import astropy.units as u
from astropy.constants import G, sigma_sb, c

import matplotlib.pyplot as plt
class BinaryBays:
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
                while P_buddy.value > 10**6:
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
        #i_buddy = np.pi/2*u.rad
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
        # rv_buddy_foo = rv_buddy.copy()
        for n in range(len(err)):
            rv_buddy[n] += np.random.normal(0, err[n].value) * u.km/u.s
        # data = RVData(t=date, rv=rv_buddy, stddev=err)
        # ax = data.plot()
        # ax.plot(date,rv_buddy_foo,'x', color = 'red')
        # ax.set_xlabel("Time [JD]")
        # ax.set_ylabel("RV [km/s]")
        # plt.show()
        # plt.close()
        return rv_buddy, err, buddy_dict
    # TODO: Get rid of the reduce option. I never use it and it just causes more
    #       problems then it's worth.

    def Real_Data_Beta(self,a = 0.3, b = 0.61):
        """
        Finds all of the beta values for the real data.
        :return: beta_data
        """
        beta_data = []
        for J in range(len(self.AAS_TABLE)):
            jit = self._jitter(J, a, b)
            err = np.sqrt(self.AAS_TABLE['RADIAL_ERR'][J]**2 + jit**2)
            beta = self._beta(self.AAS_TABLE['RADIALV'][J], err)
            beta_data.append(beta)
        beta_data = [item for sublist in beta_data for item in sublist]
        return beta_data

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

    def beta_hist(self, beta_array, bins = np.arange(0, 5.5, 0.5)):
        betarray = [b if b < max(bins) else max(bins)-0.01 for b in beta_array]
        betahist = np.histogram(betarray, bins)
        return betahist

    def beta_mod_dict(self,m_min, period, fraction, loops, jitter, a = 0.3, b = 0.6, bins = np.arange(0, 5.5, 0.5)):
        beta_dict = {x: [] for x in bins}
        loop = 0
        while loop < loops:
            print('Starting loop {}'.format(loop))
            beta = self.BetaMaster(m_min, period, fraction, 1, jitter, a, b)
            beta = self.beta_hist(beta, bins)
            for n in range(len(bins) - 1):
                beta_dict[beta[1][n]].append(beta[0][n])
            print('Done with loop {}'.format(loop))
            loop += 1
        return beta_dict

    def _beta_mod_stats(self, beta_array):
        """
        Finds the first few statistical moments for the beta array found from a model. These values are then used to find
        the skewed normal distribution that fits the model.

        Input
        ----------
        beta_array:  Array of beta values. This should be from the simulated beta distribution, for only one of
                            the bins.
        Outputs
        ----------
        mu:     Mean value
        sigma:  Standard deviation
        gamma:  Skew
        """
        mu = np.mean(beta_array)
        sigma = np.std(beta_array)
        gamma = sps.skew(beta_array)
        return mu, sigma, gamma

    def posterior(self, m_min, period,bf, loops, jitter, a=0.3, b=0.6, bins = np.arange(0, 5.5, 0.5)):
        """
        Finds the posteriod for one model set

        Inputs
        ----------
        m_min:      Minimum mass I want to consider, needs to be in Jupyter masses
        period:     Period distribution, needs to be an array with three element
                    period[0]:  Either 'U' or 'L' for Uniform or Lognormal
                    period[1]:  Mean value if log normal, or minimum value if uniform
                    period[2]:  STD if lognormal, or maximum value if uniform
        bf:         Binary fraction can range from 0 to 1
        jitter:     Boolian, either true or false, if I want to consider jitter or not
        a:          model parameter if I'm using jitter
        b:          Model parameter for jitter
        bins:       Bins I want to use for my histogram distribuitions

        Output
        ----------
        post:       posterior probability for the given set of parameters.

        """
        rd_beta = self.Real_Data_Beta(a, b)  # Get the beta values for the real data.
        rd_beta_hist = self.beta_hist(rd_beta, bins = bins)  # Histogram of the real beta data.

        rd_beta_hist_dict = {}
        for n in range(len(rd_beta_hist[0])):
            rd_beta_hist_dict[rd_beta_hist[1][n]] = [rd_beta_hist[0][n]]

        # Dictionary object of all the synthetic beta values.
        #print("Starting synthetic run {}".format(bf))
        sy_beta_dict = self.beta_mod_dict(m_min, period, bf, loops, jitter, a, b, bins)
        #print("Done with {}".format(bf))
        likelihood = 1
        for key in rd_beta_hist_dict.keys():
            # While working on this I want to make plots of each bin Just so I can watch this happen.
            mean, std, skew = self._beta_mod_stats(sy_beta_dict[key])  # Find the mean, std, and skew of the bin


            lh_foo = sps.norm.pdf(rd_beta_hist_dict[key], mean, std)
            #print(np.log(lh_foo),lh_foo, mean, std, skew, rd_beta_hist_dict[key])
            # If any one of these terms are zero then we should just exit out right away.
            # if lh_foo == 0:
            #     likelihood *= lh_foo
            #     return likelihood

            # plt.figure(figsize = (8,6))
            # plt.hist(sy_beta_dict[key], density = True)
            # xfoo = np.linspace(min(sy_beta_dict[key]) - min(sy_beta_dict[key])*0.1, max(sy_beta_dict[key])+max(sy_beta_dict[key])*0.1, 1000)
            # plt.plot(xfoo, sps.norm.pdf(xfoo, mean, std))
            # plt.vlines(rd_beta_hist_dict[key], 0, 0.01)
            # plt.title('PDF for Bin {} and Fraction {}, a = {}, b = {}'.format(key, bf, a, b))
            # plt.show()
            # plt.close()


            likelihood *= lh_foo


        return likelihood

    def Binary_Fraction(self, m_min, period,bf, loops, jitter, a=0.3, b=0.6, bins = np.arange(0, 5.5, 0.5)):
        post = {}
        for i in bf:
            #print("Starting {}".format(i))
            post_foo = self.posterior(m_min, period, i, loops, jitter, a, b, bins)
            post[i] = post_foo
            #print("Done with fraction {}".format(i))
        return  post
