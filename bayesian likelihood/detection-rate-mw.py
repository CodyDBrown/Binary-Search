import os
# import datetime
# import sys
import numpy as np
import warnings
from astropy.io import fits
# from astropy.utils.exceptions import AstropyWarning
# import time
# import shutil
# import subprocess
# import logging
# # from scipy.signal import convolve2d
# from scipy.ndimage.filters import convolve
# import astropy.stats
# import struct
# import tempfile

from astropy.table import Table, Column, vstack

import astropy.units as u
import emcee
from table_convert import table_convert

from lnprob import lnprob
from second_reduce import second_reduce
from error_with_jitter import error_with_jitter
from binary_fraction import binary_fraction

import pickle as rick

if __name__ == "__main__":

    mw = rick.load(open("milky-way-allStar-rv.pck", "rb"))
    # lmc = fits.getdata('all-average-lmc.fits')

    mw = second_reduce(Table(mw))
    mw['RADIALV'].unit = u.km / u.s
    mw['RADIAL_ERR'].unit = u.km / u.s

    # mw_err_j = error_with_jitter(mw['RADIAL_ERR'], 0.3, 0.61, mw['LOGG'])
    #
    # mw_detection_rate = binary_fraction(mw['RADIALV'],
    #                                      mw["RADIAL_ERR"])
    # mw_detection_rate_j = binary_fraction(mw['RADIALV'], mw)

    ndim, nwalkers = 5, 10
    pos = [np.array([0.5, 0, 0.61,3, 2]) + 0.3* np.random.randn(ndim) for i in range(nwalkers)]

    threads = 4
    nsteps = 10

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=threads, args=(10, mw, 100*u.jupiterMass,))

    sampler.run_mcmc(pos, nsteps)
    del sampler.pool
    # pickle "sampler"
    outfile = "detection_mw1.pck"
    if os.path.exists(outfile): os.remove(outfile)
    rick.dump(sampler, open(outfile, "wb"))
    # sampler = pickle.load( open( "save.p", "rb" ) )