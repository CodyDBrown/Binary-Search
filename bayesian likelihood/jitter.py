import astropy.units as u


def jitter(c1, c2, logg):
    """
    Finds the theoretical 'jitter' that will be added to the noise
    :param c1: y-intersept in log space, when a = 0.3 it matches the value in Hecker/Troup
    :param c2: slope of the jitter in log space, when b = 0.61 it matches the value in Hecker/Troup
    :param logg: Surface gravity
    :return: jitter
    """
    return 10 ** (c1 + c2 * logg) * u.km/u.s
