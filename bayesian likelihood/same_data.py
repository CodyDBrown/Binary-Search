from astropy.table import Table
import numpy as np


def same_dat(table1, table2, temp_lim=250, logg_lim=0.2, feh_lim=0.2):
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
        if len(gd) == 1:
            table3_foo.add_row(table1_foo[gd[0]])
            table1_foo.remove_row(gd[0])
            # print(len(table1_foo))
        elif len(gd) > 2:
            rand = np.random.randint(0, len(gd))
            table3_foo.add_row(table1_foo[gd[rand]])
            table1_foo.remove_row(gd[rand])
        else:
            rr.append(j)

    table2_foo.remove_rows(rr)

    return table3_foo, table2_foo
