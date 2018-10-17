from astropy.table import Table, Column
from astropy.io import fits
import numpy as np
import astropy.units as u
from astropy.constants import G, sigma_sb, c

def Table_Convert(table):
    """
    I need to convert the string rows back into lists and object types, so that I wan use it for Binary_Fraction
    :param table:
    :return:
    """

    # Turn those columns into type object
    table_foo = Table(table, copy = True)
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