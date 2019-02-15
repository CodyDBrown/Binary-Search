from astropy.table import Table
def second_reduce(table):
    rr = []
    for N in range(len(table)):
        if len(table['RADIALV'][N]) <= 5:
            rr.append(N)

    reduced_table = table.copy()
    reduced_table.remove_rows(rr)

    return reduced_table