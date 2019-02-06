import numpy as np
def lnprior(bf, a, b, mu, sigma):
    if 0.01 < bf < 1 and -1 < a < 1 and 0 < b < 2 and 1 < mu < 5 and 0.5 < sigma < 5:
        return 0
    return -np.inf