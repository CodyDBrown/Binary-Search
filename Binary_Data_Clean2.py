from astropy.table import Table, Column
from astropy.io import fits
import numpy as np
import astropy.units as u
from astropy.constants import G, sigma_sb, c
class Binary_Data_Clean2:
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
        ALL_AVERAGE = fits.open(self.all_average_path)
        ALL_VISIT = fits.open(self.all_visit_path)
        ISO = fits.open(self.iso_path)
        all_average_data = ALL_AVERAGE[1].data
        all_visit_data = ALL_VISIT[1].data
        iso_data = ISO[1].data
        return all_average_data, all_visit_data, iso_data

    def cuts(self, all_visit_data, SNR_cut = True):
        """
        There are a lot of points that I want to get rid of. This is kind of a
        catch all to get rid of those points. The main things it removes are
        velocity points that are nonphysical (faster than light), and data points
        with low SNR

        Inputs
        ----------
        all_visit_data: This should be a data object from a fits file
        SNR_cut:        Boolian, if I want to also remove low SNR data points

        Output
        ----------
        all_visit_data: Returns the all_visit_data object we passed as an input
                        but with the bad data points removed
        """
        #Get rid of any velocity points that are inf
        all_visit_data = all_visit_data[np.isfinite(all_visit_data['VHELIO'])]

        #Able to togle weather or not I want to remove SNR points
        if SNR_cut:
            all_visit_data = all_visit_data[all_visit_data['SNR'] >= 5] #Only keep SNR greater than..
        return all_visit_data[all_visit_data['VHELIO'] < 10**4] #Keep velocities les than light speed ish

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
        """ NOT SURE IF I NEED THIS SECTION STILL OR NOT???
        # Down the road we'll also have problems with AK and K so we need to get rid of any AK values that are -9999.9999
        AAS_RG['AK_WISE'][AAS_RG['AK_WISE'] < 0  ] = np.nan #replaces bad values with nan
        AAS_RG = AAS_RG[np.isfinite(AAS_RG['AK_WISE'])]

        AAS_RG['AK_TARG'][AAS_RG['AK_TARG'] < 0  ] = np.nan #replaces bad values with nan
        AAS_RG = AAS_RG[np.isfinite(AAS_RG['AK_TARG'])]
        return AAS_RG
        """

    def iso_fit(self,all_average_data,iso_data, limit = 0.1):
        """
        Isochrone fitting. I have to turn all_average_data into an Astropy table
        just because I don't know how to do it as the data object. I should figure
        out how to do it because the data object is less memory intensive and seams
        to run a little faster

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
        iso_meanM = np.zeros(len(all_average_data))
        iso_medianM = np.zeros(len(all_average_data))
        iso_stdM = np.zeros(len(all_average_data))

        iso_meanL = np.zeros(len(all_average_data))
        iso_medianL = np.zeros(len(all_average_data))
        iso_stdL = np.zeros(len(all_average_data))

        # For each row in all_average_data, find isochrone rows that have similar
        # values, and make a list of all of those entries.
        for j in range(len(all_average_data)):
            star_teff = all_average_data['TEFF'][j]
            star_logg = all_average_data['LOGG'][j]
            star_feh  = all_average_data['FE_H'][j]

            gd, = np.where(( (np.abs(iso_data['LOGTE']-np.log10(star_teff)) < limit) &
                    (np.abs(iso_data['LOGG']-star_logg) < limit) &
                    (np.abs(iso_data['FEH']-star_feh) < limit) ) )

            # Take the list of isochrone fittings and find the mena, mediant, and
            # deviation for the values they gave back. If they weren't any good
            # firts then make it nan
            if len(gd) > 0:
                iso_meanM[j] = np.mean(iso_data['MASS'][gd])
                iso_medianM[j] = np.median(iso_data["MASS"][gd])
                iso_stdM[j] = np.std(iso_data["MASS"][gd])

                iso_meanL[j] = np.mean(iso_data['LOGL'][gd])
                iso_medianL[j] = np.median(iso_data["LOGL"][gd])
                iso_stdL[j] = np.std(iso_data["LOGL"][gd])
            else:
                iso_meanM[j] = np.nan
                iso_medianM[j] = np.nan
                iso_stdM[j] = np.nan

                iso_meanL[j] = np.nan
                iso_medianL[j] = np.nan
                iso_stdL[j] = np.nan
        # Now add on the mass luminocity and radius values
        all_average_data['ISO_MEANM'] = iso_meanM*u.solMass
        all_average_data["ISO_MEDIANM"] = iso_medianM * u.solMass
        all_average_data['ISO_STDM'] = iso_stdM * u.solMass

        all_average_data['ISO_MEANL'] = iso_meanL * u.solLum
        all_average_data["ISO_MEDIANL"] = iso_medianL * u.solLum
        all_average_data['ISO_STDL'] = iso_stdL * u.solLum

        iso_meanR = np.sqrt( all_average_data['ISO_MEANL'] / (4 * np.pi * sigma_sb * (all_average_data["TEFF"]*u.K)**4 ) ).to(u.solRad)
        all_average_data['ISO_MEANR'] = iso_meanR
        all_average_data = all_average_data[np.isfinite(all_average_data['ISO_MEANM'])]
        return all_average_data

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
        ALL_AVERAGE_SIMPLIFIED_foo = Table(all_average_data)
        all_visit_data = Table(all_visit_data)

        RV_Column = Column(name = 'RADIALV', data = np.ones(len(ALL_AVERAGE_SIMPLIFIED_foo)), dtype = 'U111') #Right now filling it with ones just as a place holder
        ERR_Column = Column(name = 'RADIAL_ERR', data = np.ones(len(ALL_AVERAGE_SIMPLIFIED_foo)), dtype = 'U111')
        DATE_Column = Column(name = 'RADIAL_DATE', data = np.ones(len(ALL_AVERAGE_SIMPLIFIED_foo)), dtype = 'U141')
        ALL_AVERAGE_SIMPLIFIED_foo.add_columns([RV_Column,ERR_Column,DATE_Column], [0,0,0])
        all_visit_data.sort(['APOGEE_ID', 'JD'])
        count = 0
        for ID in ALL_AVERAGE_SIMPLIFIED_foo['APOGEE_ID']:
            rv_foo = np.array(all_visit_data['VHELIO'][all_visit_data['APOGEE_ID'] == ID])
            rv_err_foo = np.array(all_visit_data['VRELERR'][all_visit_data['APOGEE_ID'] == ID])
            rv_err_foo = np.array([.1 if n < 0.1 else n for n in rv_err_foo])
            date_foo = np.array(all_visit_data['JD'][all_visit_data['APOGEE_ID'] == ID])

            ALL_AVERAGE_SIMPLIFIED_foo['RADIALV'][count] = str(rv_foo).strip('[]')
            ALL_AVERAGE_SIMPLIFIED_foo['RADIAL_ERR'][count] = str(rv_err_foo).strip('[]')
            ALL_AVERAGE_SIMPLIFIED_foo['RADIAL_DATE'][count] = str(date_foo).strip('[]')
            count += 1
        return ALL_AVERAGE_SIMPLIFIED_foo

    # Lastly I want to be able to split the data table into the LMC and the SMC. One program with a flag should be able to do it
    def mc_cut(self, ALL_AVERAGE, flag):
        if flag != "L" and flag != 'S':
            return print("ERROR: flag needs to be a string, either 'L' or 'S'")
        ALL_AVERAGE_foo = Table(ALL_AVERAGE, copy = True)
        rows_2_remove = []
        for n in range(len(ALL_AVERAGE_foo)):
            if ALL_AVERAGE_foo['FIELD'][n][0] != flag:
                rows_2_remove.append(n)
        ALL_AVERAGE_foo.remove_rows(rows_2_remove)

        return ALL_AVERAGE_foo

    def Table_Convert(table):
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