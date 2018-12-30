import numpy as np
import scipy.stats as sps
import datetime as dt
from scipy import optimize
import astropy.units as u
from astropy.constants import G, sigma_sb, c

"""
IMPORTANT NOTE!!!!!!
You must have scipy version 1.2.0 or better. The way newtons method is set up it used a new vectorization that only is
not available in 1.1.0
"""

"""
This third version is suposed to use pre-build tables with a large number of 
"""


class BinaryBays3:
    def __init__(self, path_to_synth_file, path_to_data_file):
        self.synth = fits.open(path_to_synth_file)
        self.data = fits.open(path_to_data_file)

    def _jitter(self, N, a, b):
        """
        Modle the jitter using different paramiters using an equation 10^(a - b*logg). The default values give a curve
        really similar to the hecker equation 2*0.015**(1/3*logg)
        :param N: Row of the star you're on
        :param a:
        :param b:
        :return: jit
        """
        return 10 ** (a - b * self.['LOGG'][N])

    def fake_rv_binary(self, N, m_min, period, jitter, a, b):
        """
        After I have a set of paramiters form the buddy_values I want to make a fake Radial Velocity Measurment
        based on those values and the Primary Stars Values.
        """

        buddy_dict = self.buddy_values(N, m_min, period)

        date = self.AAS_TABLE['RADIAL_DATE'][N]

        if jitter:
            jit = self._jitter(N, a, b)

            err = (np.sqrt(self.AAS_TABLE['RADIAL_ERR'][N] ** 2 + jit ** 2)) * u.km / u.s
        else:
            err = self.AAS_TABLE['RADIAL_ERR'][N] * u.km / u.s

        M = 2 * np.pi / buddy_dict['P'].value * date - buddy_dict['phi'].value

        # print('M values', M, 'eccentricity', buddy_dict['e'].value,'\n')

        ec = buddy_dict['e'].value

        E0 = M.copy()
        E = optimize.newton(self._g, E0, fprime=self._gp, args=(M, ec), maxiter=50)

        # assert all(self._g(E, M, ec) < 1e-7)
        f0 = E.copy()
        f = optimize.newton(self._h, f0, fprime=self._hp, args=(E, buddy_dict['e'].value))

        rv_buddy = buddy_dict['K'] * (
                    np.cos(buddy_dict['w'].value + f) + buddy_dict['e'].value * np.cos(buddy_dict['w']))
        rv_buddy += self.AAS_TABLE['VHELIO_AVG'][N] * u.km / u.s

        rv_buddy += err * np.random.normal(0, 1,
                                           len(rv_buddy))  # This should be the same as the for loop in the comment
        # block below. This will do it slightly faster

        return rv_buddy, err, buddy_dict

    def Real_Data_Beta(self, a=0.3, b=0.61):
        """
        Finds all of the beta values for the real data.
        :return: beta_data
        """
        beta_data = []
        for J in range(len(self.AAS_TABLE)):
            jit = self._jitter(J, a, b)
            err = np.sqrt(self.AAS_TABLE['RADIAL_ERR'][J] ** 2 + jit ** 2)
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
        assert len(beta) == len(rv) * (len(rv) - 1) / 2  # Again sanity check to make sure it worked
        return beta

    # Need to Find a beta distribution for Model data.
    def _betamod(self, N, m_min, period, b_fraction, jitter, a, b):
        binary_rng = np.random.uniform()
        if binary_rng < b_fraction:  # Then we're in a binary
            fake_rv, fake_err, buddy_dict = self.fake_rv_binary(N, m_min, period, jitter, a, b)
            beta = self._beta(fake_rv, fake_err)
            return beta
        else:  # Solo star
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

    def BetaMaster(self, m_min, period, b_fraction, loop, jitter, a=0.3, b=0.61):
        foo_exit = 0
        beta_list = []
        while foo_exit < loop:
            for N in range(len(self.AAS_TABLE)):
                one_star_beta = self._betamod(N, m_min, period, b_fraction, jitter, a, b)
                beta_list.append(one_star_beta)
            foo_exit += 1
        flat_beta = [item for sublist in beta_list for item in sublist]
        return flat_beta

    def beta_hist(self, beta_array, bins=np.arange(0, 5.5, 0.5)):
        betarray = [b if b < max(bins) else max(bins) - 0.01 for b in beta_array]
        betahist = np.histogram(betarray, bins)
        return betahist

    def beta_mod_dict(self, m_min, period, fraction, loops, jitter, a=0.3, b=0.6, bins=np.arange(0, 5.5, 0.5)):
        beta_dict = {x: [] for x in bins}
        loop = 0
        while loop < loops:
            print('Starting loop {}'.format(loop))
            beta = self.BetaMaster(m_min, period, fraction, 1, jitter, a, b)
            beta = self.beta_hist(beta, bins)
            for n in range(len(bins) - 1):
                beta_dict[beta[1][n]].append(beta[0][n])
            #  print('Done with loop {}'.format(loop))
            loop += 1
        print("Done with fraction ", fraction, dt.datetime.now())
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
        return mu, sigma

    def lnlike(self, m_min, period, bf, loops, jitter, a=0.3, b=0.6, bins=np.arange(0, 5.5, 0.5)):
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
        rd_beta_hist = self.beta_hist(rd_beta, bins=bins)  # Histogram of the real beta data.

        rd_beta_hist_dict = {}
        for n in range(len(rd_beta_hist[0])):
            rd_beta_hist_dict[rd_beta_hist[1][n]] = [rd_beta_hist[0][n]]

        # Dictionary object of all the synthetic beta values.
        sy_beta_dict = self.beta_mod_dict(m_min, period, bf, loops, jitter, a, b, bins)

        ln_ans = 0

        # TO DO: Get rid of this.
        for key in rd_beta_hist_dict.keys():
            # While working on this I want to make plots of each bin Just so I can watch this happen.
            mean, std, = self._beta_mod_stats(sy_beta_dict[key])  # Find the mean, std, and skew of the bin

            if std == 0:  # Make sure I don't devide by zero
                std = 1e-8
            lnl = -1 / 2 * ((rd_beta_hist_dict[key] - mean) ** 2 / std ** 2 + np.log(2 * np.pi * std ** 2))

            ln_ans += lnl

        return ln_ans

    # Include the jitter???

    def lnprior(self, bf, a, b):
        if 0.01 < bf < 1 and -10 < a < 0.5 and 0 < b < 2:
            return 0
        return -np.inf

    def lnprob(self, theta, m_min, period, loops, jitter, bins):
        bf, a, b = theta
        lp = self.lnprior(bf, a, b)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(m_min, period, bf, loops, jitter, a, b, bins)

    """
    Start with more starts and short steps, to get a good guess of what the paramiter space looks like, then go into more 
    depth when I see where the minimums are. 

    500 - 10,000 final steps when all is said and done. 

    How many Threads? 
    I'm doing 8 as of right now. (N - 1)*2 where N is the max number of threds.

    """
