from astropy.table import Table
import numpy as np


def table_convert(table):
    """
    I need to convert the string rows back into lists and object types, so that I wan use it for Binary_Fraction
    :param table:
    :return:
    """

    # Turn those columns into type object
    table_foo = Table(table, copy=True)
    table_foo[table_foo.columns[0].name] = table_foo[table_foo.columns[0].name].astype(object)
    table_foo[table_foo.columns[1].name] = table_foo[table_foo.columns[1].name].astype(object)
    table_foo[table_foo.columns[2].name] = table_foo[table_foo.columns[2].name].astype(object)

    for n in range(len(table_foo)):
        rv = table_foo[table_foo.columns[0].name][n].split()
        rv = [float(x) for x in rv]

        rerr = table_foo[table_foo.columns[1].name][n].split()
        rerr = [float(x) for x in rerr]

        rdate = table_foo[table_foo.columns[2].name][n].split()
        rdate = [float(x) for x in rdate]

        table_foo[table_foo.columns[0].name][n] = np.array(rv)
        table_foo[table_foo.columns[1].name][n] = np.array(rerr)
        table_foo[table_foo.columns[2].name][n] = np.array(rdate)
    return table_foo
