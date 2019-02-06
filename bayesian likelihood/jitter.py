import astropy.units as u
def jitter(a, b, logg):
    """
    Finds the theoretical 'jitter' that will be added to the noise
    :param a: y-intersept in log space, when a = 0.3 it matches the value in Hecker/Troup
    :param b: slope of the jitter in log space, when b = 0.61 it matches the value in Hecker/Troup
    :param logg: Surface gravity
    :return: jitter
    """
    return 10 ** (a - b * logg) * u.km/u.s

