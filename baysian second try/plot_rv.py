import matplotlib.pyplot as plt
import numpy as np
def plot_rv(rv, date, error, title):
    fig = plt.figure(figsize=(5.6, 5.5))
    plt.errorbar(date, rv, yerr = error, fmt = '.')
    plt.hlines(np.mean(rv), date[0], date[-1])
    plt.title(title)
    return fig
