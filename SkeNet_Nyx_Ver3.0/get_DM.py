from pathlib import Path
from skimage.transform import resize
import time
import h5py
import numpy as np
import torch

from astropy.table import Table

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

#from data_loader import *
#from model import *
#from train import *
#from val import *

# Path and data file name
folder   = Path.cwd().parent / 'Nyx'
nbody_name  = 'N1024_L20_z2.4-nbody.h5'
hydro_name = 'N1024_L20_z2.4-hydro.h5'

f_nbody = h5py.File(folder/nbody_name, 'r')

data = np.array(f_nbody['native_fields']['dark_matter_density'][:])
data = resize(data, (169,169,169))
np.save(folder/'deltaDM_Nyx_L20_N160_z2.4.npy', data)

data = np.array(f_nbody['native_fields']['particle_vx'][:])
data = resize(data, (169,169,169))
np.save(folder/'vx_DM_Nyx_L20_N160_z2.4.npy', data)

data = np.array(f_nbody['native_fields']['particle_vy'][:])
data = resize(data, (169,169,169))
np.save(folder/'vy_DM_Nyx_L20_N160_z2.4.npy', data)

data = np.array(f_nbody['native_fields']['particle_vz'][:])
data = resize(data, (169,169,169))
np.save(folder/'vz_DM_Nyx_L20_N160_z2.4.npy', data)