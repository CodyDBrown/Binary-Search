from astropy.table import Table, Column
from astropy.io import fits
import numpy as np
import astropy.units as u
from astropy.constants import G, sigma_sb, c
class Binary_Data_Clean:
    """
    Goal of this class is it have all of the data simplifying stuff
    in one place. The process should go as follows,
    1.  Import the fits files and make them an astropy Tables. There should be an all visit table, and an 
        all averaged tables.
    
    2.  Make a cut so that we only have stars with at least 5 observations. This should eliminate rows in
        both the all visit and all average tables
    
    3.  Add the RV measurments, RV errors, and the dates observed as a column of arrays to the all average
        table. This way I don't need to have both tables around anymore I can just use the averaged table.
        
    4.  Make a cut so that I only have stars that fall into the red giant branch.
    
    5.  Use Isochrone tables to find the mass and luminocity of the stars based off the effective surface 
        tempurature, surface gravity and iron ratios
    """
    
    #Initialize, with two paths to the files I want to read in
    def __init__(self, all_average_path, all_visit_path, iso_path):
        self.all_average_path = all_average_path
        self.all_visit_path = all_visit_path
        self.iso_path = iso_path
    #Reads in the tables 
    def load_tables(self):
        ALL_AVERAGE = Table(fits.getdata(self.all_average_path,1))
        ALL_VISIT = Table(fits.getdata(self.all_visit_path, 1))
        ISO = Table(fits.getdata(self.iso_path,1))

        return ALL_AVERAGE, ALL_VISIT, ISO
    
    #There are some points that have unphysical velocities
    def too_fast(self,ALL_VISIT):
        ALL_VISIT = ALL_VISIT[np.isfinite(ALL_VISIT['VHELIO'])]
        remove_too_fast = []
        dumby_too_fast = list(range(0,len(ALL_VISIT)))
        
        for y in dumby_too_fast:
            if ALL_VISIT['VHELIO'][y] > 10**4:
                remove_too_fast.append(dumby_too_fast[y])
        ALL_VISIT.remove_rows(remove_too_fast)
        return ALL_VISIT
        
    #Makes the cuts for number of visits must be greater than 5
    def nvisits_cut(self,ALL_AVERAGE,ALL_VISIT):
        #I want to make copies of the table so that when I make changes to them it doesn't affect the original tables
        ALL_AVERAGE_SIMPLIFIED = Table(ALL_AVERAGE, copy = True)
        ALL_VISIT_SIMPLIFIED = Table(ALL_VISIT, copy = True)
        rows_remove = []
        for n in range(len(ALL_AVERAGE_SIMPLIFIED)):
            if ALL_AVERAGE_SIMPLIFIED['NVISITS'][n] < 5:
                rows_remove.append(n)
        ALL_AVERAGE_SIMPLIFIED.remove_rows(rows_remove)
        
        holder = []
        
        #TO DO: Figure out how to get rid of this nested for loop. This is the bottle neck for the whole process
        for n in range(len(ALL_VISIT_SIMPLIFIED)):
            for m in ALL_AVERAGE_SIMPLIFIED['APOGEE_ID']:
                if ALL_VISIT_SIMPLIFIED["APOGEE_ID"][n] == m:
                    holder.append(n)
                    break
        #Holder is a list of all the rows I want to keep. So I need to get rid of all the other ones
        remove_allvis = []
        dumby_allvis = list(range(0,len(ALL_VISIT_SIMPLIFIED)))
        for x in dumby_allvis:
            if x not in holder:
                remove_allvis.append(dumby_allvis[x])
                
        ALL_VISIT_SIMPLIFIED.remove_rows(remove_allvis)
        return ALL_AVERAGE_SIMPLIFIED, ALL_VISIT_SIMPLIFIED
    
    #Make a separate red giant table.
    def rg_cut(self,AllAvg):
        AllAvgSimp_foo = Table(AllAvg, copy = True)
        #Get rid of bad tempurature and log(g) values. 
        AllAvgSimp_foo['TEFF'][AllAvgSimp_foo['TEFF'] < 0 ] = np.nan #replaces bad values with nan
        AllAvgSimp_foo['LOGG'][AllAvgSimp_foo['LOGG'] < 0 ] = np.nan
        AllAvgSimp_foo['FE_H'][AllAvgSimp_foo['FE_H'] < -100] = np.nan


        AllAvgSimp_foo = AllAvgSimp_foo[np.isfinite(AllAvgSimp_foo['TEFF'])] #gets rid of nan's
        AllAvgSimp_foo = AllAvgSimp_foo[np.isfinite(AllAvgSimp_foo['LOGG'])]
        AllAvgSimp_foo = AllAvgSimp_foo[np.isfinite(AllAvgSimp_foo['FE_H'])]
        # I only want to look at the Red Giant group right now so I'm going to ignore
        AAS_RG = Table(AllAvgSimp_foo, copy = True)
        AAS_RG['TEFF'][AAS_RG['TEFF'] >= 5500  ] = np.nan 
        AAS_RG['LOGG'][AAS_RG['LOGG'] >= 3.7 + 0.1*AAS_RG['FE_H'] ] = np.nan

        #Now get rid of nan's
        AAS_RG = AAS_RG[np.isfinite(AAS_RG['TEFF'])] #gets rid of nan's
        AAS_RG = AAS_RG[np.isfinite(AAS_RG['LOGG'])]

        # Down the road we'll also have problems with AK and K so we need to get rid of any AK values that are -9999.9999
        AAS_RG['AK_WISE'][AAS_RG['AK_WISE'] < 0  ] = np.nan #replaces bad values with nan
        AAS_RG = AAS_RG[np.isfinite(AAS_RG['AK_WISE'])]

        AAS_RG['AK_TARG'][AAS_RG['AK_TARG'] < 0  ] = np.nan #replaces bad values with nan
        AAS_RG = AAS_RG[np.isfinite(AAS_RG['AK_TARG'])]
        return AAS_RG
        
    
    #Adding the RV, err, and dates to ALL_AVERAGE_SIMPLIFIED. This should be the last one you
    #run because you don't want to waist time adding RV values for stars you will end up ignoring
    def rv_table_add(self, ALL_AVERAGE_SIMPLIFIED, ALL_VISIT_SIMPLIFIED):
        RV_Column = Column(name = 'RADIALV', data = np.ones(len(ALL_AVERAGE_SIMPLIFIED)), dtype = 'object') #Right now filling it with ones just as a place holder
        ERR_Column = Column(name = 'RADIAL_ERR', data = np.ones(len(ALL_AVERAGE_SIMPLIFIED)), dtype = 'object')
        DATE_Column = Column(name = 'RADIAL_DATE', data = np.ones(len(ALL_AVERAGE_SIMPLIFIED)), dtype = 'object')
        
        ALL_AVERAGE_SIMPLIFIED_foo = Table(ALL_AVERAGE_SIMPLIFIED, copy = True)
        ALL_AVERAGE_SIMPLIFIED_foo.add_columns([RV_Column,ERR_Column,DATE_Column], [0,0,0])
        
        start = 0
        end = 0
        loop = 0
        name = 0
        #Just in case I want to organize by APOGEE_ID and JD
        ALL_VISIT_SIMPLIFIED.sort(['APOGEE_ID', 'JD'])
        
        for n in ALL_AVERAGE_SIMPLIFIED_foo['NVISITS']:
            start = end
            end += n
            
            #Read in the dates the object was observed
            Date = ALL_VISIT_SIMPLIFIED['JD'][start:end] - ALL_VISIT_SIMPLIFIED['JD'][start]
            Date_Array = np.array(Date)
            
            RV = ALL_VISIT_SIMPLIFIED['VHELIO'][start:end]
            RV_Array = np.array(RV)
            
            err = ALL_VISIT_SIMPLIFIED['VRELERR'][start:end]
            
            err = [.1 if n < 0.1 else n for n in err]
            err_Array = np.array(err)
            
            #Now Add them to the all averaged simplified table
            ALL_AVERAGE_SIMPLIFIED_foo['RADIALV'][name] = RV_Array
            ALL_AVERAGE_SIMPLIFIED_foo['RADIAL_ERR'][name] = err_Array
            ALL_AVERAGE_SIMPLIFIED_foo['RADIAL_DATE'][name] = Date_Array
            name += 1
        return ALL_AVERAGE_SIMPLIFIED_foo
    
    ### ISOCHRON FITTING ###
    
    def iso_fit(self,ALL_AVERAGE,ISO):
        iso_meanM = np.zeros(len(ALL_AVERAGE))
        iso_medianM = np.zeros(len(ALL_AVERAGE))
        iso_stdM = np.zeros(len(ALL_AVERAGE))
        
        iso_meanL = np.zeros(len(ALL_AVERAGE))
        iso_medianL = np.zeros(len(ALL_AVERAGE))
        iso_stdL = np.zeros(len(ALL_AVERAGE))
        
        for j in range(len(ALL_AVERAGE)):
            star_teff = ALL_AVERAGE['TEFF'][j]
            star_logg = ALL_AVERAGE['LOGG'][j]
            star_feh  = ALL_AVERAGE['FE_H'][j]
            
            gd, = np.where(( (np.abs(ISO['LOGTE']-np.log10(star_teff)) < 0.1) &
                    (np.abs(ISO['LOGG']-star_logg) < 0.1) &
                    (np.abs(ISO['FEH']-star_feh) < 0.1) ) )
            
            # This if loop is to check that any good values got found. If the array is empty then when i find mean and the like
            # it gives me some warnings about dividing by zero. 
            if len(gd) > 0:
                iso_meanM[j] = np.mean(ISO['MASS'][gd])
                iso_medianM[j] = np.median(ISO["MASS"][gd])
                iso_stdM[j] = np.std(ISO["MASS"][gd])

                iso_meanL[j] = np.mean(ISO['LOGL'][gd])
                iso_medianL[j] = np.median(ISO["LOGL"][gd])
                iso_stdL[j] = np.std(ISO["LOGL"][gd])
            else:
                iso_meanM[j] = np.nan
                iso_medianM[j] = np.nan
                iso_stdM[j] = np.nan

                iso_meanL[j] = np.nan
                iso_medianL[j] = np.nan
                iso_stdL[j] = np.nan
        #Now add on the mass luminocity and radius values
        ALL_AVERAGE['ISO_MEANM'] = iso_meanM*u.solMass
        ALL_AVERAGE["ISO_MEDIANM"] = iso_medianM * u.solMass
        ALL_AVERAGE['ISO_STDM'] = iso_stdM * u.solMass

        ALL_AVERAGE['ISO_MEANL'] = iso_meanL * u.solLum
        ALL_AVERAGE["ISO_MEDIANL"] = iso_medianL * u.solLum
        ALL_AVERAGE['ISO_STDL'] = iso_stdL * u.solLum 

        iso_meanR = np.sqrt( ALL_AVERAGE['ISO_MEANL'] / (4 * np.pi * sigma_sb * (ALL_AVERAGE["TEFF"]*u.K)**4 ) ).to(u.solRad)
        ALL_AVERAGE['ISO_MEANR'] = iso_meanR
        ALL_AVERAGE = ALL_AVERAGE[np.isfinite(ALL_AVERAGE['ISO_MEANM'])]
        return ALL_AVERAGE
    
    #Lastly I want to be able to split the data table into the LMC and the SMC. One program with a flag should be able to do it
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