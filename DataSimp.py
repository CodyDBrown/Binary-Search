from astropy.table import Table, Column
from astropy.io import fits
import numpy as np
import astropy.units as u
from astropy.constants import G, sigma_sb, c


class DataSimp:
    """
    Goal of this class is it have all of the data simplifying stuff in one place.
    The process should go as follows,
    1.  Import the fits files and make them an astropy Tables. There should be
        an all visit table, and an all averaged tables.

    2.  Make a cut so that we only have stars with at least 5 observations. This
        should eliminate rows in both the all visit and all average tables

    3.  Add the RV measurments, RV errors, and the dates observed as a column of
        arrays to the all average table. This way I don't need to have both tables
        around anymore I can just use the averaged table.

    4.  Make a cut so that I only have stars that fall into the red giant branch.

    5.  Use Isochrone tables to find the mass and luminocity of the stars based
        off the effective surface tempurature, surface gravity and iron ratios
    """

    """
    # TODO: Make my own all average table.

    """
    # Initialize, with two paths to the files I want to read in
    def __init__(self, all_average_path, all_visit_path, iso_path):
        self.all_average_path = all_average_path
        self.all_visit_path = all_visit_path
        self.iso_path = iso_path

    def get_data(self):
        all_average_data = fits.getdata(self.all_average_path)
        all_visit_data = fits.getdata(self.all_visit_path)
        iso_data = fits.getdata(self.iso_path)
        return all_average_data, all_visit_data, iso_data

    def avg_column_keep(self, all_average, columns=None):
        if columns is None:
            columns = ['APOGEE_ID', 'FIELD',
                       'NVISITS', 'SNR',
                       'VHELIO_AVG', 'VERR',
                       'TEFF', 'TEFF_ERR',
                       'LOGG','LOGG_ERR',
                       "FE_H","FE_H_ERR",
                       "RV_TEFF", "RV_LOGG",
                       "RV_FEH"]
        data = []
        for c in columns:
            c_foo = np.array(all_average[c])
            data.append(c_foo)
            print("done with ", c)
        t = Table(data, names=columns)
        return t

    def vis_column_keep(self, all_visit, columns=None):
        if columns is None:
            columns = ['APOGEE_ID', "MJD", "JD", "SNR", "VHELIO", "VRELERR"]
        data = []
        for c in columns:
            c_foo = np.array(all_visit[c])
            data.append(c_foo)
            print("done with ", c)
        t = Table(data, names = columns)
        return t

    def cuts(self, all_visit_data, snr_cut=True):
        """
        There are a lot of points that I want to get rid of. This is kind of a catch all to get rid of those points.
        The main things it removes are velocity points that are nonphysical (faster than light), and data points with
        low SNR

        Inputs
        ----------
        all_visit_data: This should be a data object from a fits file
        snr_cut:        Boolian, if I want to also remove low SNR data points

        Output
        ----------
        all_visit_data: Returns the all_visit_data object we passed as an input
                        but with the bad data points removed
        """
        # Get rid of any velocity points that are inf
        all_visit_data = all_visit_data[np.isfinite(all_visit_data['VHELIO'])]

        # Able to togle weather or not I want to remove SNR points
        if snr_cut:
            all_visit_data = all_visit_data[np.isfinite(all_visit_data['SNR'])]
            all_visit_data = all_visit_data[all_visit_data['SNR'] >= 5]  # Only keep SNR greater than..
        return all_visit_data[all_visit_data['VHELIO'] < 10 ** 4]  # Keep velocities les than light speed ish


    def nvisits_cut(self, all_average_data, n_cut):
        """
        Modifies the averaged data to only have stars with a sertain number of
        visits

        Inputs
        ----------
        all_average_data:   Data object from the all averaged fits file
        n_cut               Minimum number of visits I want to consider

        Outputs
        ----------
        all_average_data:   Returns the input data object but all rows where
                            NVISITS was less than n_cut are removed
        """

        all_average_data = all_average_data[all_average_data['NVISITS'] >= n_cut]
        return all_average_data

    # Make a separate red giant table.
    def rg_cut(self, all_average_data):
        """
        Removes stars in the all average data set so that we're left with stars
        that fall into the Red Giant Branch (RGB)
        """
        # Get rid of any nonphysical values
        all_average_data = all_average_data[all_average_data['TEFF'] > 0]
        all_average_data = all_average_data[all_average_data['LOGG'] > 0]
        all_average_data = all_average_data[all_average_data['FE_H'] > -100]

        #Taken from Troup et. al (2016)
        all_average_data = all_average_data[all_average_data['TEFF'] <= 5500]
        all_average_data = all_average_data[all_average_data['LOGG'] <= 3.7 + 0.1*all_average_data['FE_H']]

        return all_average_data

    def iso_fit(self, all_average_data, iso_data):
        """
        Isochrone fitting. I have to turn all_average_data into an Astropy table
        just because I don't know how to do it as the data object. I should figure
        out how to do it because the data object is less memory intensive and seams
        to run a little faster.
        CHANGES TO V3: I want to find the 'best fit' for the isochrone table. But the 'best fit' is hard to
        define. So
        I'm going to start by setting the limit of the fit equal to zero, look for an exact match. Then slowly expand the
        range that we look at untill we find a fit. When we expand the range we look at we want to expand each paramiter
        based on the error in that measurment. We'll examd the limit for each paramiters by half of it's
        recorded error.
        Hopefully we just find one fit, and it makes physical since.
        CHANGES TO V4: Want to find the distance in log space to the observed values to the isochrone values.
        This goal
        is to have a better way of finding the best fit and then keep all stars so that we never throw away
        any values.
        I would like it if we still took into account the size of the errors.
        Inputs
        ----------
        all_average_data:   All averaged data object
        iso_data:           isochrone data object
        Output
        ----------
        all_average_data:   All averaged data TABLE, that has added columns for
                            mass, luminocity and radius
        """
        # Turn the data object into an Astropy Table
        all_average_data = Table(all_average_data)

        # Make some zero arrays that I'll fill with stellar paramiters
        iso_M = np.zeros(len(all_average_data))

        iso_L = np.zeros(len(all_average_data))

        iso_check_dist = np.zeros(len(all_average_data))
        # Only want to look at stars that have an age greater than 1 giga year (billion)
        iso_data_foo = iso_data[iso_data["AGE"] >= 1e9]

        # For each row in all_average_data, find isochrone rows that have similar
        # values, and make a list of all of those entries.
        for j in range(len(all_average_data)):
            star_teff = all_average_data['TEFF'][j]
            star_logg = all_average_data['LOGG'][j]
            star_feh = all_average_data['FE_H'][j]



            # For time reasons I want to cut down the isochrone table into a smaller table for the max range I want to
            # look at. Then I can expand the search range I want to look at but then I'm only looking at this
            # one table
            # and not waisting time looking over the entire table each time.
            sig = 2
            gd_foo, = np.where((
                    (np.abs((10 ** iso_data_foo['LOGTE']) - star_teff) <= sig * all_average_data['TEFF_ERR'][j]) &
                    (np.abs(iso_data_foo['LOGG'] - star_logg) <= sig * all_average_data['LOGG_ERR'][j]) &
                    (np.abs(iso_data_foo['FEH'] - star_feh) <= sig * all_average_data['FE_H_ERR'][j])))

            while len(gd_foo) < 1:
                gd_foo, = np.where((
                        (np.abs((10 ** iso_data_foo['LOGTE']) - star_teff) <= sig * all_average_data['TEFF_ERR'][j]) &
                        (np.abs(iso_data_foo['LOGG'] - star_logg) <= sig * all_average_data['LOGG_ERR'][j]) &
                        (np.abs(iso_data_foo['FEH'] - star_feh) <= sig * all_average_data['FE_H_ERR'][j])))
                iso_small_table = iso_data_foo[gd_foo]
                sig += 1
            iso_small_table = Table(iso_data_foo[gd_foo])

            # Find the 'distance' from the star to the nearest isochrone neighbor
            iso_distance = np.sqrt((iso_small_table['LOGTE'] - np.log10(star_teff)) ** 2 +
                                   (iso_small_table['LOGG'] - star_logg) ** 2 +
                                   (iso_small_table['FEH'] - star_feh) ** 2)

            col_dist = Column(name='dist', data=iso_distance)

            iso_small_table.add_column(col_dist)
            iso_small_table.sort('dist')
            iso_M[j] = iso_small_table['MASS'][0]
            iso_L[j] = 10 ** (iso_small_table["LOGL"][0])
            iso_check_dist[j] = iso_small_table['dist'][0]
            #         print(iso_M, iso_L, iso_small_table['dist'][0:4])

            iso_small_table.remove_column("dist")

            if j in np.arange(0, len(all_average_data), 100):
                print("Done with %f" % j)

        # Now add on the mass luminocity and radius values
        all_average_data['ISO_MASS'] = iso_M * u.solMass

        all_average_data['ISO_LUM'] = (iso_L) * u.solLum

        iso_meanR = np.sqrt(all_average_data['ISO_LUM'] /
                            (4 * np.pi * sigma_sb * (all_average_data["TEFF"] * u.K) ** 4)).to(u.solRad)
        all_average_data['ISO_RAD'] = iso_meanR
        all_average_data['ISO_FIT_DIST'] = iso_check_dist
        #     all_average_data = all_average_data[np.isfinite(all_average_data['ISO_MEANM'])]
        #     all_average_data = all_average_data[all_average_data["ISO_MEANM"] > 0]
        return all_average_data


    """
    troup_fits and troup_errors are old function's I don't really use anymore, but I'm keeping them around in case I need
    them in the future.
    """
    def troup_fits(self,all_average_data,):
        aad = Table(all_average_data)
        aad = aad[aad['K'] < 99] # Bad values of 'K' magnitude are stored as 99. Make sure we get rid of those.
        A_K = [aad['AK_TARG'][n] if aad['AK_TARG'][n] > 0 else aad['AK_WISE'][n] for n in range(len(aad)) ]

        A_K = [n if n > 0 else 0 for n in A_K]
        K0  = aad['K'] - A_K

        # Find the Bolometric Correction
        BC_K = (6.8 - 0.2 * aad['FE_H'])*(3.96 - np.log10(aad['TEFF']))


        m_bol = K0 + BC_K # Observed bolometic magnitude

        # Now I need to look at distance between LMC and SMC
        distances = [49.97e3 if aad['FIELD'][n][0] == 'L' else 60.6e3 for n in range(len(aad))] * u.pc

        M_bol = m_bol - 5 * np.log10(distances.value) + 5 # Absolute magnitues

        L = 10**(-0.4*(M_bol - 4.77)) * u.solLum
        R = np.sqrt( L / (4 * np.pi * sigma_sb * (aad['TEFF']*u.K)**4)).to(u.solRad)
        M = ((10**(aad['LOGG']))*(u.cm/u.s**2) * R**2 / G).to(u.solMass)


        # Add these as columns in the table
        aad['m_bol'] = m_bol
        aad['M_bol'] = M_bol
        aad['K0'] = K0
        aad['A_K'] = A_K
        aad['BC_K'] = BC_K

        aad['T_LUM'] = L
        aad['T_RADIUS'] = R
        aad['T_MASS'] = M

        return aad

    def troup_errors(self, aad):
        """
        Finds the errors for the troup fits

        :param aad:
        :return: all_average_data but with error columns added
        """
        K0_ERR = aad['K_ERR']

        # Find the Bolometric Correction Errors
        dbdf = -0.2*(3.96-np.log10(aad['TEFF']))
        dbdt = -(6.8 - 0.2*aad['FE_H']) / (aad['TEFF']*np.log(10))
        BC_K_ERR = np.sqrt((dbdf*aad['FE_H_ERR'])**2 + (dbdt*aad['TEFF_ERR'])**2)

        m_bol_ERR = K0_ERR + BC_K_ERR

        distances = [49.97e3 if aad['FIELD'][n][0] == 'L' else 60.6e3 for n in range(len(aad))] * u.pc
        distances_ERR = ([1.126e3 if aad['FIELD'][n][0] == 'L' else 1e3 for n in range(len(aad))] * u.pc)

        M_bol_ERR = np.sqrt((1*m_bol_ERR)**2 + (-5/(distances*np.log(10)) * distances_ERR)**2)


        L_ERR = abs(10 ** (-0.4 * (aad['M_bol'] - 4.77))* -0.4 * np.log(10) *M_bol_ERR)* u.solLum


        dRdL = np.sqrt(1/(4*np.pi*sigma_sb*(aad['TEFF']*u.K)**4)) * 0.5 * aad['T_LUM']**(-1/2)/(u.solLum**1.5)  # This last bit is a little hacky
                                                                                                                # But for some reason the -1/2 power
                                                                                                                # Doesn't do anything to the unit on T_LUM
        dRdT = np.sqrt(aad['T_LUM']/ (4*np.pi*sigma_sb)) * -2 * (aad['TEFF']*u.K)**-3

        R_ERR = (np.sqrt( (dRdL*L_ERR)**2 + (dRdT * aad['TEFF_ERR']*u.K)**2)).to(u.solRad)

        foo_log = aad['LOGG'] + aad['LOGG_ERR']
        dMdR = 2 * (10 ** (aad['LOGG'])) * (u.cm / u.s ** 2) * aad['T_RADIUS'] / G
        dMdG = (10 ** (aad['LOGG'])) * (u.cm / u.s ** 2) * np.log(10) * aad['T_RADIUS'] ** 2 / G * u.solRad

        M_ERR = (np.sqrt((dMdG * aad['LOGG_ERR']) ** 2 + (dMdR * R_ERR) ** 2)).to(u.solMass)

        # Add these as columns in the table
        aad['m_bol_ERR'] = m_bol_ERR
        aad['M_bol_ERR'] = M_bol_ERR
        aad['K0_ERR'] = K0_ERR
        aad['BC_K_ERR'] = BC_K_ERR

        aad['T_LUM_ERR'] = L_ERR
        aad['T_RADIUS_ERR'] = R_ERR
        aad['T_MASS_ERR'] = M_ERR

        return aad

    def rv_table_add(self, all_average_data, all_visit_data):
        """
        Add the radial velocity measurments to the all_average_data set. That way
        you only need the once all_average_data to do binary fraction and don't
        need to look back and forth between the different tables

        Inputs
        ----------
        all_average_data:   Averaged data
        all_visit_data:     Individual data points

        Output
        ----------
        """
        # Make the data objects Tables
        # ALL_AVERAGE_SIMPLIFIED_foo = Table(all_average_data)

        RV_Column = Column(name = 'RADIALV',
                           data = np.ones(len(all_average_data)),
                           dtype = 'object') #Right now filling it with ones just as a place holder
        ERR_Column = Column(name = 'RADIAL_ERR',
                            data = np.ones(len(all_average_data)),
                            dtype = 'object') # Why a string limit of 151? Because that's when I stopped getting errors.
        DATE_Column = Column(name = 'RADIAL_DATE',
                             data = np.ones(len(all_average_data)),
                             dtype = 'object')

        all_average_data.add_columns([RV_Column,ERR_Column,DATE_Column], [0,0,0])

        count = 0
        for ID in all_average_data['APOGEE_ID']:
            mask = all_visit_data['APOGEE_ID'] == ID
            rv_foo = np.array(all_visit_data['VHELIO'][mask])
            rv_err_foo = np.array(all_visit_data['VRELERR'][mask])
            rv_err_foo = np.array([.1 if n < 0.1 else n for n in rv_err_foo])
            date_foo = np.array(all_visit_data['JD'][mask])

            all_average_data['RADIALV'][count] = rv_foo
            all_average_data['RADIAL_ERR'][count] = rv_err_foo
            all_average_data['RADIAL_DATE'][count] = date_foo
            count += 1


        return all_average_data

    # Lastly I want to be able to split the data table into the LMC and the SMC. One program with a flag should be able to do it
    # def mc_cut(self, ALL_AVERAGE, flag):
    #     if flag != "L" and flag != 'S':
    #         return print("ERROR: flag needs to be a string, either 'L' or 'S'")
    #     ALL_AVERAGE_foo = Table(ALL_AVERAGE, copy = True)
    #     rows_2_remove = []
    #     for n in range(len(ALL_AVERAGE_foo)):
    #         if ALL_AVERAGE_foo['FIELD'][n][0] != flag:
    #             rows_2_remove.append(n)
    #     ALL_AVERAGE_foo.remove_rows(rows_2_remove)
    #
    #     return ALL_AVERAGE_foo

    def mc_cut(self, ALL_AVERAGE, flag):
        ALL_AVERAGE_foo = Table(ALL_AVERAGE, copy=True)
        rows_2_remove = []
        x = len(flag)
        for n in range(len(ALL_AVERAGE_foo)):
            if ALL_AVERAGE_foo['FIELD'][n][0:x] != flag:
                rows_2_remove.append(n)
        ALL_AVERAGE_foo.remove_rows(rows_2_remove)

        return ALL_AVERAGE_foo
    def Table_Convert(self, table):
        """
        I need to convert the string rows back into lists and object types, so that I wan use it for Binary_Fraction
        :param table:
        :return:
        """

        # Turn those columns into type object
        table_foo = Table(table, copy=True)
        table_foo['RADIALV'] = table_foo['RADIALV'].astype(object)
        table_foo['RADIAL_ERR'] = table_foo['RADIAL_ERR'].astype(object)
        table_foo['RADIAL_DATE'] = table_foo['RADIAL_DATE'].astype(object)

        for n in range(len(table_foo)):
            rv = table_foo['RADIALV'][n].split()
            rv = [float(x) for x in rv]

            rerr = table_foo['RADIAL_ERR'][n].split()
            rerr = [float(x) for x in rerr]

            rdate = table_foo['RADIAL_DATE'][n].split()
            rdate = [float(x) for x in rdate]

            table_foo['RADIALV'][n] = np.array(rv)
            table_foo['RADIAL_ERR'][n] = np.array(rerr)
            table_foo['RADIAL_DATE'][n] = np.array(rdate)
        return table_foo

