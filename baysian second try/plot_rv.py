import matplotlib.pyplot as plt
import numpy as np
def plot_rv(rv, date, error):
    plt.figure(figsize = (10,10))
    plt.errorbar(date, rv, yerr = error, fmt = '.')
    plt.hlines(np.mean(rv), date[0], date[-1])
    plt.show()
    plt.close()