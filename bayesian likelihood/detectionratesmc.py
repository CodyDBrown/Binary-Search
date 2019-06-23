import os
import datetime
import sys
import numpy as np
import warnings
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import time
import shutil
import subprocess
import logging
# from scipy.signal import convolve2d
from scipy.ndimage.filters import convolve
import astropy.stats
import struct
import tempfile

from astropy.table import Table, Column, vstack
import scipy.stats as sps
import astropy.units as u
import emcee
from table_convert import table_convert

from lnprob import lnprob
from second_reduce import second_reduce
from error_with_jitter import error_with_jitter
from binary_fraction import binary_fraction

import pickle

if __name__ == "__main__":

    lmc = fits.getdata('all-average-smc.fits')
    lmc = table_convert(lmc)
    lmc = second_reduce(Table(lmc))
    lmc['RADIALV'].unit = u.km / u.s
    lmc['RADIAL_ERR'].unit = u.km / u.s

    lmc_err_j = error_with_jitter(lmc['RADIAL_ERR'], 0.3, 0.61, lmc['LOGG'])

    lmc_detection_rate = binary_fraction(lmc['RADIALV'],
                                         lmc["RADIAL_ERR"])
    lmc_detection_rate_j = binary_fraction(lmc['RADIALV'], lmc_err_j)

    ndim, nwalkers = 5, 14
    pos = [np.array([0.5, 0, 0.61, 3, 2]) + 0.3 * np.random.randn(ndim) for i in range(nwalkers)]

    threads = 8
    nsteps = 1000

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=threads, args=(50, lmc, 100 * u.jupiterMass,))

    sampler.run_mcmc(pos, nsteps)
    del sampler.pool
    # pickle "sampler"
    outfile = "detection_smc-2019-03-10.pck"
    if os.path.exists(outfile): os.remove(outfile)
    pickle.dump(sampler, open(outfile, "wb"))
    # sampler = pickle.load( open( "save.p", "rb" ) )