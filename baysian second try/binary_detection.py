from scipy.stats import chi2
import chi_sq_mean
def binary_detection(rv, error):
    """
    Determines if the rv and error are in a binary or not.
    Inputs
    ----------
    rv:     Radial velocity
    error:  errors for the measurments
    :return: Boolian, true if the rv and error gives a reasonable binary, false if not
    """

    assert len(rv) == len(error) # Sanity check to make sure I have the same number of data points for rv and error
    chi_squared = chi_sq_mean.chi_sq_mean(rv, error)

    p_value = 1 - chi2.cdf(chi_squared, len(rv) - 1)

    return p_value < 0.05
