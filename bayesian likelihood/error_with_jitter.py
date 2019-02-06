from jitter import jitter

def error_with_jitter(error, a, b, logg):
    """
    Adds jitter noise to the error
    :param error:
    :param a:
    :param b:
    :param logg:
    :return:
    """
    jit = jitter(a, b, logg)

    return (error**2 + jit.value**2)**0.5 # For some reason np.sqrt() wasn't working