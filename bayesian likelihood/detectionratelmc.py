import os
# import datetime
# import sys
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

    lmc = pickle.load(open("/home/cody/Binary Search/all-average-lmc.pck", "rb" ) )

    lmc['RADIALV'].unit = u.km / u.s
    lmc['RADIAL_ERR'].unit = u.km / u.s

    ndim, nwalkers = 5, 14
    threads = 7
    nsteps = 1000

    # A = [0.6, 0.3, -0.1]
    # B = [1, 0.61, 0]
    # for a in A:
    #     for b in B:
    pos = [np.array([0.5, 0.3, 0.6, 3, 2]) + 0.3* np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    threads=threads,
                                    args=(15,
                                          lmc,
                                          100*u.jupiterMass,
                                          )
                                    )

    sampler.run_mcmc(pos, nsteps)
    del sampler.pool
    # pickle "sampler"
    outfile = "detection_lmc-2019-02-27.pck"
    if os.path.exists(outfile): os.remove(outfile)
    pickle.dump(sampler, open(outfile, "wb"))
    # sampler = pickle.load( open( "save.p", "rb" ) )