

import os
import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table

from enigma.tpe.skewer_tools import random_skewers
from enigma.tpe.utils import calc_eosfit2, make_tau_skewers

# Generate Random skewers for calibrating UVB
Nran = 10  # 1000 ## Hack to make it run quickly
seed = 1
DMAX=3000.0
zstr='z2.4'
sim_path = '~/Documents/skewers/skewers_ML/Nyx'
out_path = '~/Documents/skewers/skewers_ML/Nyx/' + zstr + '/'
hdf5file=sim_path +'N1024_L20_'+ zstr + '-hydro.h5'
ranovtfile = out_path + 'ran_skewers_' + zstr + '_OVT.fits'
rantaufile = out_path + 'ran_skewers_' + zstr + '_OVT_tau.fits'
ret = random_skewers(Nran, hdf5file, ranovtfile, seed)  # ,Ng,rand)

# Read in skewers
params = Table.read(ranovtfile, hdu=1)
skewers = Table.read(ranovtfile, hdu=2)

Nskew = len(skewers)
Ng = (skewers['ODEN'].shape)[1]

# Fit equation of state (EOS)
oden = skewers['ODEN'].reshape(Nskew * Ng)
T = skewers['T'].reshape(Nskew * Ng)
(logT0, gamma) = calc_eosfit2(oden, T, -0.5, 0.0)
params['EOS-logT0'] = logT0
params['EOS-GAMMA'] = gamma

GAMMA_NOW = 0.1

retval = make_tau_skewers(params, skewers, rantaufile, DMAX,
                          GAMMA_UVB=GAMMA_NOW, RESCALE=False, IMPOSE_EOS=False)

# Now read in the skewers
params = Table.read(rantaufile, hdu=1)
skewers = Table.read(rantaufile, hdu=2)

# Generate Random skewers for calibrating UVB
#Nran = 10  # 1000 ## Hack to make it run quickly
#seed = 1
#params, skewers = lya_joe('z05', sim_path, out_path, Nran, seed)
