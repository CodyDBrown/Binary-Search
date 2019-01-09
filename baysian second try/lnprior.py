def lnprior(bf, a, b, mu, sigma):
    if 0.01 < bf < 1 and -10 < a < 0.5 and 0 < b < 2 and 0 < mu < 5 and 1 < sigma < 
        return 0
    return -np.inf