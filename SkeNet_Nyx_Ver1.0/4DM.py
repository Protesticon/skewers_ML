from pathlib import Path
from skimage.transform import resize
import time
import h5py
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from skimage.transform import resize

def get_index(x, y, cofm, crange, DM_size):
    lx, ly = crange
    Nskew = len(x)
    x_range = ((x.reshape(Nskew,1)+(np.arange(lx)-(lx-1)/2))%DM_size).astype('int')
    y_range = ((y.reshape(Nskew,1)+(np.arange(ly)-(ly-1)/2))%DM_size).astype('int')

    cx = x_range.repeat(ly).reshape(Nskew,lx,ly).transpose(0,1,2)
    cy = y_range.repeat(lx).reshape(Nskew,ly,lx).transpose(0,2,1)
    
    return cx, cy#tuple([cx, cy])

# Path and data file name
folder   = Path.cwd().parent / 'Nyx'
nbody_name  = 'N1024_L20_z2.4-nbody.h5'
hydro_name = 'N1024_L20_z2.4-hydro.h5'

Ng = 1024
LBOX = 31.249999999999996
spac = LBOX / 300
nblk = 20
blk  = 0 

field_list = ['dark_matter_density', 'particle_vx', 'particle_vy', 'particle_vz']
f = h5py.File(folder/nbody_name, 'r')

ax = np.round((np.linspace(spac/2, LBOX-spac/2, LBOX/spac) - LBOX/Ng/2) / (LBOX/Ng))
cofm = np.array(np.meshgrid(ax,ax,np.array([0.]))).T.reshape(-1,3) #all combinations of x and y
x, y = cofm[:,[0,1]].T
cx, cy = get_index(x, y, cofm, [45,45], 1024)
del x, y, cofm

for ii, field in enumerate(field_list):
    DM_input = np.zeros(shape=(len(cx),11,11,250))
    DM = np.array(f['native_fields'][field])
    
    for jj in range(len(cx)):
        DM_input[jj] = resize(DM[cx[jj], cy[jj]], (11,11,250))
        if (jj+1)%1000 == 0:
            print(ii, jj+1)
    np.save(folder/(field+'.npy'), DM_input)
f.close()    