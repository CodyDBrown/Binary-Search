#!/usr/bin/env python

import os
import sys
import numpy as np
import warnings
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import time
import shutil
import subprocess
import logging
#from scipy.signal import convolve2d
from scipy.ndimage.filters import convolve
import astropy.stats
import struct
import tempfile

from astropy.table import Table, Column, vstack
import scipy.stats as sps
import astropy.units as u
import emcee
from table_convert import table_convert
from BinaryBays2 import BinaryBays2
import pickle

if __name__ == "__main__":

    # Run the code

    #%run BinaryDataClean3.py

    # Inputs are paths to the fits files I'm using. First one is the allaverage file, second is the allvisit type
    # file. 3rd is the isochrone files.
    #Tables = BinaryDataClean3('/home/cody/Binary Search/2nd_Data_Set/mc_rgb_nocuts.fits','/home/cody/Binary Search/2nd_Data_Set/allVisit-t9-l31c-58247.mc.fits',
    #                                                    '/home/cody/Binary Search/parsec_decamsdss_all.fits.gz')
    aas_lmc = fits.open("all-average-lmc")
    aas_smc = fits.open('all-average-smc')
    AllAvg_SMC=table_convert(aas_smc[1].data) # Turn the strings made in rv_table_add into lists
    AllAvg_LMC=table_convert(aas_lmc[1].data)



    # Next cell is a little hacky, it's doing through everything again and making sure I only have stars with at least 5 observations.
    # When I cut out low SNR observations, the nvisits_cut, doens't hit all of the stars it should hit.

    rr = []
    for N in range(len(AllAvg_LMC)):
       if len(AllAvg_LMC['RADIALV'][N]) <= 5:
           rr.append(N)
    AllAvg_LMC.remove_rows(rr)

    rr = []
    for N in range(len(AllAvg_SMC)):
        if len(AllAvg_SMC['RADIALV'][N]) <= 5:
            rr.append(N)
    AllAvg_SMC.remove_rows(rr)

    # This uses the log likelihood, should be bug free? Set up to work well with $\textit{emcee}$
    BB2 = BinaryBays2(AllAvg_LMC)
    a = 0.1
    b = 0.61
    bob2 = BB2.lnlike(100, ['L',3.2,2.5], bf = 0.2 , loops = 10,
                      jitter = True, a = a, b = b, bins = np.arange(0,11,1))
    
    
    # This is all taken from the $\textit{emcee}$, page, but modified for what I need. Right now there are just 3 paramiters.
    # In theory I could have more, but to make sure things are working I thought I would keep it simple for now.

    ndim, nwalkers = 3, 16
    threads = 8
    nsteps = 50

    # Create the initial conditions
    pos = [np.array([0.5, 0.1, 0.61]) + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]

    # Create and run the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, BB2.lnprob, threads=threads, args = (100, ['L',3.2,2.5],10, True, np.arange(0,11,1)))
    sampler.run_mcmc(pos, nsteps)

    # pickle "sampler"
    outfile = "betasearch.pck"
    if os.path.exists(outfile): os.remove(outfile)
    pickle.dump( sampler, open(outfile, "wb" ) )
    #sampler = pickle.load( open( "save.p", "rb" ) )
