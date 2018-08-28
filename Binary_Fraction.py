from astropy.table import Table, Column, vstack
from astropy.io import fits
import numpy as np
from scipy.stats import chi2
import astropy.units as u
from astropy.constants import G, sigma_sb, c

from astropy.time import Time

from thejoker.data import RVData
from twobody.wrap import cy_rv_from_elements
from twobody.utils import ArrayProcessor
from twobody import KeplerOrbit
# TODO: One thing that I think is really hurting this project is the 'synthetic_detection_rate'
#       part of the code. What would probably be a lot more efficient would be to
#       make a cataloge of synthetic signals and then I would just have a large
#       set of fake signals that I could use over and over again. Rather than having
#       to run the synthetic_detection over and over again. Having a set synthetic
#       table I think will make my life easier
#       BUT, there's a problem with saving fits files with different numbers of
#       elements in an array. So I either make an all_visit style table, or I
#       have to save each synthetic table as it's own, run my analysis, and then
#       combine the results from all the different NVISITS values. Not sure what
#       the most efficent way to do this is.
class Binary_Fraction:
    def __init__(self,AAS_TABLE):
        self.AAS_TABLE = AAS_TABLE

    def buddy_values(self, N, m_min, period):
        """
        Makes a list of the paramiters of the secondary star for the primary star
        in row 'N' from AAS_TABLE

        Inputs
        ----------
        N: This is just the row that I want to use when looking at the AAS_TABLE
        m_min: Minimum mass I want to consider. This should be in Jupiter Masses.

        Output
        ----------
        List of the physical paramiters randomly made from the companion star.
        """
        M = self.AAS_TABLE['ISO_MEANM'][N]*u.solMass
        #Make the fake companion that we want orbiting our primary star
        m_buddy = np.random.uniform(m_min, M.to(u.jupiterMass).value)*u.jupiterMass #For reference the 1 solMas = 1047 jupMas

        #Checks what kind of period distribution we want
        if period == 'U' or period == 'u':
            P_buddy = np.random.uniform(12,1000)*u.d
        elif period == 'L' or period == 'l':
            P_buddy = np.random.lognormal(5.03,2.28)*u.d
        else:
            return print('Period flag needs to be "L" or "U" not {}'.format(period))


        if P_buddy.value < 12:
            e_buddy = 0*u.one
        else:
            e_buddy = np.random.uniform(0, 0.93)*u.one
        n_foo = (2 * np.pi) / P_buddy

        a_buddy = np.cbrt(( (G*(M + m_buddy)) / (4*np.pi**2) ) * P_buddy**2 )
        a_buddy = a_buddy.to(u.AU)

        # Also need some angles of the orbit that we would see.
        i_buddy = np.random.uniform(0, np.pi)*u.rad

         #These are some phase angles that depend on when we first see it.
        w_buddy = np.random.uniform(0, 2*np.pi)*u.rad
        phi_buddy = np.random.uniform(0, 2*np.pi)*u.rad

        #Make sure the closest point of the orbit isn't so close that we have to worry about title effects.
        r_peri = (1-e_buddy)*a_buddy
        r_peri = r_peri.to(u.solRad)

        #If the orbit is so close that we would have to consider title effects then we want to pick a different
        #set of orbital paramiters.
        in_case_of_emergency = 0 #Variable that will get us out of the loop if we're stuck forever
        while r_peri.value < 5*self.AAS_TABLE['ISO_MEANR'][N]:

            m_buddy = np.random.uniform(m_min, M.to(u.jupiterMass).value)*u.jupiterMass
            P_buddy = np.random.uniform(12,1000)*u.d
            #P_buddy = np.random.lognormal(5.03,2.28)*u.d
            if P_buddy.value < 12:
                e_buddy = 0*u.one
            else:
                e_buddy = np.random.uniform(0,.93)*u.one
            n = (2*np.pi) / P_buddy

            #From those paramiters we can use keplers law to find the semi-major axis
            a_buddy = np.cbrt(( (G*(M + m_buddy)) / (4*np.pi**2) ) * P_buddy**2 )
            a_buddy = a_buddy.to(u.AU)

            # Also need some angles of the orbit that we would see.
            i_buddy = np.random.uniform(0, np.pi)*u.rad

            #These are some phase angles that depend on when we first see it.
            w_buddy = np.random.uniform(0, 2*np.pi)*u.rad
            phi_buddy = np.random.uniform(0, 2*np.pi)*u.rad
            #Make sure the closest point of the orbit isn't so close that we have to worry about title effects.
            r_peri = (1-e_buddy)*a_buddy
            r_peri = r_peri.to(u.solRad)
            in_case_of_emergency += 1
            if in_case_of_emergency > 9:
                print("You got stuck!")
                break
        # Now I can find the "K" paramiter based on these values.
        K_buddy = (m_buddy / (M + m_buddy)) * (n_foo * a_buddy * np.sin(i_buddy)) / np.sqrt(1-e_buddy**2)
        K_buddy = K_buddy.to(u.km / u.s)
        foo_list = [m_buddy, e_buddy, P_buddy, a_buddy, i_buddy, w_buddy,
                   phi_buddy, K_buddy, self.AAS_TABLE['APOGEE_ID'][N]]

        return foo_list

    def buddy_table(self, buddy_array):
        """
        Takes the output from buddy_values and turns it into an astropy table. The goal is to have a table of
        values from the simulations. Some of
        """
        Foo_Table = Table([[False], [buddy_array[0].value], [buddy_array[1].value], [buddy_array[2].value], [buddy_array[3].value],
                           [buddy_array[4].value], [buddy_array[5].value], [buddy_array[6].value], [buddy_array[7].value],
                           [0], [buddy_array[8]]],
                          names = ('Binary','m','e','P','a','i','w','phi','K','P-value','APOGEE_ID'),
                          dtype=('b','f8','f8','f8','f8','f8','f8','f8','f8','f8','str'))
        return Foo_Table


    def fake_rv_binary(self, N, m_min, period, jitter):
        """
        After I have a set of paramiters form the buddy_values I want to make a fake Radial Velocity Measurment
        based on those values and the Primary Stars Values.
        """
        buddy_list = self.buddy_values(N,m_min, period)
        Date = self.AAS_TABLE['RADIAL_DATE'][N]

        if jitter:
            jitter_value = 2*0.015**(1/3*self.AAS_TABLE["LOGG"][N])
            err = (np.sqrt(self.AAS_TABLE['RADIAL_ERR'][N]**2 + jitter_value**2) )*u.km/u.s
        else:
            err = self.AAS_TABLE['RADIAL_ERR'][N]*u.km/u.s
        # This is some stuff that cy_rv_from_elements needs for it's time input. Not sure why but it doens't
        #work without these few lines. i.e. DO NOT TOUCH!
        t_buddy = Time(Date, format = 'mjd')

        t_buddy = t_buddy.tcb.mjd

        procb = ArrayProcessor(t_buddy)

        t_buddy, = procb.prepare_arrays()

        td0 = t_buddy[0]

        td0 = Time(td0, format = 'mjd')

        #Makes the observed radial velocity in the binaries Barrycenter.
        rv_buddy = cy_rv_from_elements(t_buddy, buddy_list[2].to(u.day).value, 1. , buddy_list[1].value, buddy_list[5].value,
                                         buddy_list[6].value, td0.tcb.mjd,
                                         anomaly_tol = 1E-10, anomaly_maxiter = 128)

        #Then we move the velocity to be in our reference frame. The extra added term at the end is to simulate
        #the fact that we wont observ the actual velocity every measurement will be off from the real value.
        rv_buddy = (buddy_list[7] * rv_buddy + self.AAS_TABLE['VHELIO_AVG'][N] * u.km/ u.s) + np.random.normal(0, self.AAS_TABLE['VERR'][N], size = len(rv_buddy)) * u.km/u.s

        return rv_buddy, err, buddy_list
    # TODO: Get rid of the reduce option. I never use it and it just causes more
    #       problems then it's worth.
    def chi_sq_mean(self, RV, err, reduce):
        """
        I want to fit some radial velocity measurments to a strait line of the average value. The logic goes
        that if the strait line fits the data well, then it's a single star floating in space. If a strait line
        fits the data poorly then there should be something there causing the deviation from the mean.

        Inputs
        ----------
        RV: Radial velocity values
        err: the error in those radial velocity values
        reduce: boolian if I want to find the reduced chi squared or not. Only
                applicable if there are more datapoints than paramiters.
        """
        chi_sq_value = np.sum((RV - np.mean(RV))**2 / (err)**2)
        if reduce:
            return chi_sq_value / (len(RV) - 1)
        else:
            return chi_sq_value
    def fake_binary_detection(self, N, m_min, period, jitter, reduce):
        """
        Uses the chi_sq_value to then find the P-Value then set the variable Binary in buddy table to be True
        or False depending on the threshold we set.
        """
        #First call the fake_rv function
        f_radial_velocity, f_err, bud_list = self.fake_rv_binary(N, m_min, period, jitter)
        #Find the chi^2 for the fake rv values
        chi_squared = self.chi_sq_mean(f_radial_velocity, f_err, reduce)
        #Find the p-value from that chi^2
        p_value = 1 - chi2.cdf(chi_squared, len(f_radial_velocity) - 1)
        #print(chi_squared, p_value)
        #Make the buddy table
        bud_table = self.buddy_table(bud_list)
        #Set the 'P-value' in the buddy table
        bud_table['P-value'] = p_value
        #Check if it's a binary or not.
        if p_value < 0.05:
            f_binary = True
        else:
            f_binary = False
        #Put the result in the table
        bud_table['Binary'] = f_binary
        """
        date = self.AAS_TABLE['RADIAL_DATE'][N]
        RV = f_radial_velocity
        data = RVData(t = date, rv = RV, stddev= f_err)
        data.plot()
        plt.title(p_value)
        plt.show()
        plt.close()
        """
        return f_radial_velocity, f_err, bud_table

    """
    Need the same thing but this time with no buddy, assumes a single star with error and decided if it's in a binary or not
    """
    def fake_solo_detection(self,N, m_min, jitter, reduce):
        jitter = 2*0.015**(1/3*self.AAS_TABLE["LOGG"][N])
        #Make some fake solo RV measurments
        solo_RV = np.random.normal(self.AAS_TABLE['VHELIO_AVG'][N], self.AAS_TABLE['VERR'][N] ,size = len(self.AAS_TABLE["RADIALV"][N]))
        #Keep the real ovserved error
        solo_err = np.sqrt(self.AAS_TABLE['RADIAL_ERR'][N]**2 + jitter**2)
        #Don't think I need the date, but my old stuff has it so I'm keeping it
        Date = self.AAS_TABLE['RADIAL_DATE'][N]
        #Finds the chi_squared and p-value
        solo_chi_squared = self.chi_sq_mean(solo_RV, solo_err, reduce)
        p_value = 1 - chi2.cdf(solo_chi_squared, len(solo_RV) - 1)
        #Check if it's a binary or not.
        if p_value < 0.05:
            solo_binary = True
        else:
            solo_binary = False
        return p_value, solo_binary

    """
    Time to put it all together
    """

    def binary_or_not(self, N, m_min, period, jitter, reduce, b_fraction):
        """
        Picks a random number and if it's below b_fraction then we run fake_binary_detection
        if it's larger than b_fraction then we run fake_solo_detection. Returns the P-value and
        the boolian 'Binary' for each run
        """
        #Pick a random number between 0 and 1.
        foo_random_number = np.random.uniform()
        if foo_random_number < b_fraction:
            foo_rv, foo_err, foo_table = self.fake_binary_detection(N,m_min, period, jitter, reduce)
            return foo_table['P-value'], foo_table['Binary']
        else:
            foo_pvalue, foo_binary = self.fake_solo_detection(N,m_min, jitter, reduce)
            return foo_pvalue, foo_binary


    def Master(self,m_min, period, jitter, reduce, b_fraction,loop):
        """
        Should just have to run this to get the detection rate for different binary
        fractions at the end of the day.
        """
        foo_exit = 0
        pvalue_list = []
        binary_bool_list = []
        while foo_exit < loop:
            for N in range(len(self.AAS_TABLE)):
                p_value, bool_binary = self.binary_or_not(N, m_min, period, jitter, reduce, b_fraction)
                pvalue_list.append(p_value)
                binary_bool_list.append(bool_binary)
            foo_exit += 1
        return pvalue_list, binary_bool_list

    def synthetic_detection_rate(self, m_min, period, jitter, reduce, b_fraction,
                                loop):
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
        #Make empty arrays for the answers to be put in.
        detection_rate = []
        p_value_result = []
        binary_result_b = []
        detection_rate_error = []
        for i in b_fraction:
            p_value_foo, binary_result_foo = self.Master(m_min, period, jitter, reduce, i, loop)

            p_value_result.append(p_value_foo)
            binary_result_b.append(binary_result_foo)
            detection_rate.append(np.count_nonzero(binary_result_foo)/len(binary_result_foo))
            error = np.sqrt(np.count_nonzero(binary_result_foo))
            det_upper = (np.count_nonzero(binary_result_foo) + error) /len(binary_result_foo)
            det_lower = (np.count_nonzero(binary_result_foo) - error) /len(binary_result_foo)
            err_foo = [det_upper, det_lower]
            detection_rate_error.append(err_foo)
        return p_value_result, binary_result_b, detection_rate, detection_rate_error

    def Synthetic_Table(self, m_min, period, jitter, reduce, loop):
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
        foo_rv, foo_err, final_table = self.fake_binary_detection(0,m_min, period, jitter, reduce)
        while foo_exit < loop:
            for N in range(len(self.AAS_TABLE)):
                foo_rv, foo_err, foo_table = self.fake_binary_detection(N,m_min, period, jitter, reduce)
                final_table = vstack([final_table, foo_table])
            print('Done with loop', foo_exit)
            foo_exit += 1
        return final_table
    """Does the binary fraction check for the real data table that was put into
    the Binary_Fraction when first initializing. I would also like to change the
    values"""
    def Real_Data_Fraction(self):
        foo_check = []
        rd_p_value_array = []
        rd_Binary_array = []
        detection_rate_error = []
        for K in range(len(self.AAS_TABLE)):
            rd_rv = self.AAS_TABLE['RADIALV'][K]
            rd_err = self.AAS_TABLE['RADIAL_ERR'][K]
            rd_jitter = 2*0.015**(1/3*self.AAS_TABLE["LOGG"][K])
            rd_err = np.sqrt(rd_err**2 + rd_jitter**2)
            rd_chi_squared = self.chi_sq_mean(rd_rv, rd_err, False)
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
