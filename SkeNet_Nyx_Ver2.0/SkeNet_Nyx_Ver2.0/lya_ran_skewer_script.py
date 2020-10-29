#!/usr/bin/env python

def lya_ran_skewer_script(spac,dmax,blk,nblk,feed=0,GAMMA_UVB=None):

    import numpy as np
    import h5py
    from astropy.table import Table
    from astropy.cosmology import FlatLambdaCDM
    from pathlib import Path
    from enigma.NyX.NyX_skewers import get_skewers
    from lya_skewer_tools import lya_skewers_manual
    from enigma.tpe.utils import make_tau_skewers
    
    sim_path = Path.home() / 'Documents/skewers/skewers_ML/Nyx'

    zstr='z2.4'
    hdf5file=sim_path / ('N1024_L20_'+ zstr + '-hydro.h5')

    manskewfile = sim_path / ('Nyx_skewers_positions_' + zstr + '_%.1fMpchspacing_calcnode.fits'%(spac))
    manovtfile = sim_path / ('Nyx_skewers_densveltemp_' + zstr + '_%.1fMpchspacing_calcnode.fits'%(spac))
    mantaufile = sim_path / ('Nyx_skewers_tau_' + zstr + '_%.1fMpchspacing_calcnode.fits'%(spac))

    
    # Open HDF5 file, and grab some relevant parameters                                                                    
    pf=h5py.File(hdf5file,'r')
    Ob0=pf['universe'].attrs['omega_b']
    Om0=pf['universe'].attrs['omega_m']
    lit_h=pf['universe'].attrs['hubble']
    z=pf['universe'].attrs['redshift']
    Ng=pf['domain'].attrs['shape'][0]
    Lbox_hMpc=pf['domain'].attrs['size'][0]
    pf.close()
    Lbox=Lbox_hMpc/lit_h
    print("Lbox=%s"%(Lbox))
    
    cosmo = FlatLambdaCDM(H0=100.0*lit_h, Om0=Om0,Ob0=Ob0)
    Hz=cosmo.H(z)
    a=1.0/(1.0 + z)
    vside = (a*Hz*Lbox).value
    aH=a*Hz.value

    print('##########Skewer positions')
    # Generate skewer positions at manual locations for the mean Ly-a forest (only z-projection)
    ret=lya_skewers_manual(manskewfile,spac,Lbox,Ng,blk,nblk)
    
    print('##########Skewer OVT')
    # Extract OVT skewers from NyX output skewers at random locations
    retval=get_skewers(ret,hdf5file,manovtfile)
    
    print('##########Read skewers')
    ## Read in skewers
    params=Table.read(manovtfile,hdu=1)
    skewers=Table.read(manovtfile,hdu=2)
    
    print('##########Skewer tau')
    retval=make_tau_skewers(params,skewers,mantaufile,dmax,RESCALE=False,GAMMA_UVB=1.)
    
    f = open(sim_path/('spectra_Nyx_'+zstr+'_%.1fmpchspacing.dat'%(spac)),'ab')
    np.savetxt(f,Table.read(mantaufile,hdu=2)['TAU'])
    f.close()
    return retval
    
    
    
splits = 40
for block in range(20):
    lya_ran_skewer_script(0.030517578124999997,1.e4,int(block+20),splits)


