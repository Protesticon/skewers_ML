# Module for proximity effect skewer creating


"""
Module for performing various tasks on NyX halo catalogs to generate 
skewers for line-of-sight and transverse proximity effects.
"""
import numpy as np
import math
from astropy.table import Table, vstack,hstack
from astropy.io import ascii
#from xastropy.xutils import xdebug as xdb


def lya_halofits(halofile,halofits,Lbox,aH,Ng,R_m,a,logMmin=0,logMmax=20,R_MAX=0,l_chunk=0):
     
    """Reads in ascii output, outputs fits file

    Parameters
    ----------
    halofile: string for ascii halofile
    halofits: string for fits output file

    Returns
    -------
    1 : upon succes
    """
    halos=Table.read(halofile,format='ascii.no_header',names=('XHALO','YHALO','ZHALO','ISTR','TECH','MASS','NCELL','VXHALO','VYHALO','VZHALO'))

    edge=max(R_MAX+l_chunk/2.,R_m/a)

    # Trim halos to be above minimum mass
    mask=(halos['MASS'] >= 10.**logMmin)
    halos=halos[mask]
    mask1=(halos['MASS'] < 10.**logMmax)
    halos=halos[mask1]
    mask2=(halos['XHALO']<Lbox-edge)
    halos=halos[mask2]
    mask3=(halos['YHALO']<Lbox-edge)
    halos=halos[mask3]
    mask4=(halos['ZHALO']<Lbox-edge)
    halos=halos[mask4]
    mask5=(halos['XHALO']>edge)
    halos=halos[mask5]
    mask6=(halos['YHALO']>edge)
    halos=halos[mask6]
    mask7=(halos['ZHALO']>edge)
    halos=halos[mask7]


    dum=np.char.array(halos['ISTR'])
    split=np.array([np.array(xi) for xi in dum.split('x')])
    I=np.array(split[:,0],dtype='int32')
    J=np.array(split[:,1],dtype='int32')
    K=np.array(split[:,2],dtype='int32')

    halos['IHALOX']=I
    halos['IHALOY']=J
    halos['IHALOZ']=K

    # Define the redshift space position
    halos['SXHALO']=0.0*I
    halos['SYHALO']=0.0*I
    halos['SZHALO']=0.0*I

    ## Create slots for W1Mpc and LOGLV
    halos['W1MPC']=0.0*I
    halos['LOGLV']=0.0*I

    # Redshift space coordinates, X
    ssumx = halos['XHALO'] + halos['VXHALO']/aH  # redshift space coordinate
    ineg  = np.where(ssumx < 0.0)[0]
    ipos =  np.where(ssumx > Lbox)[0]
    irest = np.where((ssumx >= 0.0) & (ssumx <= Lbox))[0]
    # Deal with periodicity
    if (len(ineg) > 0):
        halos['SXHALO'][ineg]=Lbox + ssumx[ineg]
    if (len(ipos) > 0):
        halos['SXHALO'][ipos]= ssumx[ipos] - Lbox
    if (len(irest)> 0):
        halos['SXHALO'][irest]=ssumx[irest]

    # Redshift space coordinates, Y
    ssumy = halos['YHALO'] + halos['VYHALO']/aH  # redshift space coordinate
    ineg  = np.where(ssumy < 0.0)[0]
    ipos =  np.where(ssumy > Lbox)[0]
    irest = np.where((ssumy >= 0.0) & (ssumy <= Lbox))[0]
    # Deal with periodicity
    if (len(ineg) > 0):
        halos['SYHALO'][ineg]=Lbox + ssumy[ineg]
    if (len(ipos) > 0):
        halos['SYHALO'][ipos]= ssumy[ipos] - Lbox
    if (len(irest)> 0):
        halos['SYHALO'][irest]=ssumy[irest]

    # Redshift space coordinates, Z
    ssumz = halos['ZHALO'] + halos['VZHALO']/aH  # redshift space coordinate
    ineg  = np.where(ssumz < 0.0)[0]
    ipos =  np.where(ssumz > Lbox)[0]
    irest = np.where((ssumz >= 0.0) & (ssumz <= Lbox))[0]
    # Deal with periodicity
    if (len(ineg) > 0):
        halos['SZHALO'][ineg]=Lbox + ssumz[ineg]
    if (len(ipos) > 0):
        halos['SZHALO'][ipos]= ssumz[ipos] - Lbox
    if (len(irest)> 0):
        halos['SZHALO'][irest]=ssumz[irest]

    # Add an unique index to each halos as an identifier
    nhalo=len(halos)
    halos['HALO_INDX']=np.arange(nhalo,dtype=int)     
    halos.write(halofits,overwrite=True)

    return halos

def select_halos(halofile,halofits,Lbox,a,aH,Ng,R_m,logMmin=0,logMmax=20,Nhalos=None):
    
    """Reads in ascii output, outputs fits file
        
        Parameters
        ----------
        halofile: string for ascii halofile
        halofits: string for fits output file
        
        Returns
        -------
        1 : upon succes
        """
    halos=Table.read(halofile,format='ascii.no_header',names=('XHALO','YHALO','ZHALO','ISTR','TECH','MASS','NCELL','VXHALO','VYHALO','VZHALO'))
    
    R_m=R_m/a # cMpc
    print("R_m=%s cMpc"%(R_m))
    
    # Trim halos to be above minimum mass
    mask=(halos['MASS'] >= 10.**logMmin)
    halos=halos[mask]
    mask1=(halos['MASS'] < 10.**logMmax)
    halos=halos[mask1]
    mask2=(halos['XHALO'] < Lbox-R_m)
    halos=halos[mask2]
    mask3=(halos['YHALO'] < Lbox-R_m)
    halos=halos[mask3]
    mask4=(halos['ZHALO'] < Lbox-R_m)
    halos=halos[mask4]
    mask5=(halos['XHALO'] > R_m)
    halos=halos[mask5]
    mask6=(halos['YHALO'] > R_m)
    halos=halos[mask6]
    mask7=(halos['ZHALO'] > R_m)
    halos=halos[mask7]

    dum=np.char.array(halos['ISTR'])
    split=np.array([np.array(xi) for xi in dum.split('x')])
    I=np.array(split[:,0],dtype='int32')
    J=np.array(split[:,1],dtype='int32')
    K=np.array(split[:,2],dtype='int32')
    halos['IHALO']=I
    halos['JHALO']=J
    halos['KHALO']=K

    # Define the redshift space position
    halos['SXHALO']=0.0*I
    halos['SYHALO']=0.0*I
    halos['SZHALO']=0.0*I

    ## Create slots for W1Mpc and LOGLV
    halos['W1MPC']=0.0*I
    halos['LOGLV']=0.0*I

    # Redshift space coordinates, X
    ssumx = halos['XHALO'] + halos['VXHALO']/aH  # redshift space coordinate
    ineg  = np.where(ssumx < 0.0)[0]
    ipos =  np.where(ssumx > Lbox)[0]
    irest = np.where((ssumx >= 0.0) & (ssumx <= Lbox))[0]
    # Deal with periodicity
    if (len(ineg) > 0):
        halos['SXHALO'][ineg]=Lbox + ssumx[ineg]
    if (len(ipos) > 0):
        halos['SXHALO'][ipos]= ssumx[ipos] - Lbox
    if (len(irest)> 0):
        halos['SXHALO'][irest]=ssumx[irest]

    # Redshift space coordinates, Y
    ssumy = halos['YHALO'] + halos['VYHALO']/aH  # redshift space coordinate
    ineg  = np.where(ssumy < 0.0)[0]
    ipos =  np.where(ssumy > Lbox)[0]
    irest = np.where((ssumy >= 0.0) & (ssumy <= Lbox))[0]
    # Deal with periodicity
    if (len(ineg) > 0):
        halos['SYHALO'][ineg]=Lbox + ssumy[ineg]
    if (len(ipos) > 0):
        halos['SYHALO'][ipos]= ssumy[ipos] - Lbox
    if (len(irest)> 0):
        halos['SYHALO'][irest]=ssumy[irest]

    # Redshift space coordinates, Z
    ssumz = halos['ZHALO'] + halos['VZHALO']/aH  # redshift space coordinate
    ineg  = np.where(ssumz < 0.0)[0]
    ipos =  np.where(ssumz > Lbox)[0]
    irest = np.where((ssumz >= 0.0) & (ssumz <= Lbox))[0]
    # Deal with periodicity
    if (len(ineg) > 0):
        halos['SZHALO'][ineg]=Lbox + ssumz[ineg]
    if (len(ipos) > 0):
        halos['SZHALO'][ipos]= ssumz[ipos] - Lbox
    if (len(irest)> 0):
        halos['SZHALO'][irest]=ssumz[irest]

    # Add an unique index to each halos as an identifier
    nhalo=len(halos)
    halos['HALO_INDX']=np.arange(nhalo,dtype=int)     

    
    if not (Nhalos==None):
        halos=halos[0:Nhalos]

    halos.write(halofits,overwrite=True)
    return halos


def lya_skewers_tpe(halos,skewerfile,Nskew,R_MIN,R_MAX,LBOX,Ng,rand,XPROJ=False,YPROJ=False,Ng_chunk=0):
     
    XHALO=halos['XHALO']
    YHALO=halos['YHALO']
    ZHALO=halos['ZHALO']

    # Project all three cyclic permutations (x,y),(z,x),(y,z)

    # xy skewers, project along z
    X,Y,IX,IY,BEGIN,END,XHALO_out,YHALO_out,ZHALO_out,IHALO,JHALO,KHALO=lya_skewers_halos(XHALO, YHALO, ZHALO, Nskew, R_MIN, R_MAX, LBOX, Ng,rand,Ng_chunk)
                           #print('length KHALO BEGIN END %s %s %s '%(len(KHALO), len(BEGIN), len(END)))
    """
    # Eliminate first N_halos rows, because they are the skewers halos positions, which we do not need.
    N_halos=len(halos['XHALO'])
    print("Nhalos=%s"%(N_halos))
    X=X[N_halos:]
    Y=Y[N_halos:]
    IX=IX[N_halos:]
    IY=IY[N_halos:]
    """
    FDUM=0.0*X
    IDUM=0*IX
    PROJ_SIGN=0*IX + 1 # positive projection
    PROJ_AXIS=0*IX + 2 # projected along z
    skew_xy = Table([X,Y,FDUM,IX,IY,IDUM,PROJ_AXIS,PROJ_SIGN,XHALO_out,YHALO_out,ZHALO_out,IHALO,JHALO,KHALO,BEGIN,END]
                 ,names=('XSKEW','YSKEW','ZSKEW','ISKEWX','ISKEWY','ISKEWZ'
                         ,'PROJ_AXIS','PROJ_SIGN','XHALO','YHALO','ZHALO','IHALO','JHALO','KHALO','BEGIN','END'),masked=True)
    skew_xy['ISKEWZ'].mask[:]=True
    skew_xy['ZSKEW'].mask[:]=True

    if((YPROJ==False)and(XPROJ==False)):
        skewers=vstack([skew_xy])

    if (YPROJ==True):
        BEGIN=JHALO-(Ng_chunk/2)*np.ones(len(JHALO))
        END=JHALO+(Ng_chunk/2)*np.ones(len(JHALO))
        # zx skewers, project along y
        Z,X,IZ,IX,BEGIN,END,ZHALO_out,XHALO_out,YHALO_out,KHALO,IHALO,JHALO=lya_skewers_halos(ZHALO, XHALO, YHALO, Nskew, R_MIN, R_MAX, LBOX, Ng,rand,Ng_chunk)
        FDUM=0.0*X
        IDUM=0*IX
        PROJ_SIGN=0*IX + 1 # positive projection
        PROJ_AXIS=0*IX + 1 # projected along y
        skew_zx = Table([X,FDUM,Z,IX,IDUM,IZ,PROJ_AXIS,PROJ_SIGN,XHALO_out,YHALO_out,ZHALO_out,IHALO,JHALO,KHALO,BEGIN,END]
                      ,names=('XSKEW','YSKEW','ZSKEW','ISKEWX','ISKEWY','ISKEWZ'
                         ,'PROJ_AXIS','PROJ_SIGN','HALO_INDX','XHALO','YHALO','ZHALO','IHALO','JHALO','KHALO','BEGIN','END'),masked=True)
        skew_zx['ISKEWY'].mask[:]=True
        skew_zx['YSKEW'].mask[:]=True
        # concatenate the tables. The projection direction will be masked as a missing
        # value in the respective skewers                                                                                                                                                  
        if(XPROJ==False):
               skewers=vstack([skew_xy,skew_zx])

    if (XPROJ==True):
        BEGIN=IHALO-(Ng_chunk/2)*np.ones(len(IHALO))
        END=IHALO+(Ng_chunk/2)*np.ones(len(IHALO))
        # yz skewers, project along x
        Y,Z,IY,IZ,BEGIN,END,YHALO_out,ZHALO_out,XHALO_out,JHALO,KHALO,IHALO=lya_skewers_halos(YHALO, ZHALO, XHALO, Nskew, R_MIN, R_MAX, LBOX, Ng,rand,Ng_chunk)
        FDUM=0.0*Y
        IDUM=0*IY
        PROJ_SIGN=0*IY + 1 # positive projection
        PROJ_AXIS=0*IY     # projected along x
        skew_yz = Table([FDUM,Y,Z,IDUM,IY,IZ,PROJ_AXIS,PROJ_SIGN,XHALO_out,YHALO_out,ZHALO_out,IHALO,JHALO,KHALO,BEGIN,END]
                      ,names=('XSKEW','YSKEW','ZSKEW','ISKEWX','ISKEWY','ISKEWZ'
                              ,'PROJ_AXIS','PROJ_SIGN','XHALO','YHALO','ZHALO','IHALO','JHALO','KHALO','BEGIN','END'),masked=True)
        skew_yz['ISKEWX'].mask[:]=True
        skew_yz['XSKEW'].mask[:]=True
        # concatenate the tables. The projection direction will be masked as a missing
        # value in the respective skewers

    skewers.write(skewerfile,overwrite=True)
    return True

def build_halo_cube(halos,skewerfile):
    
    XHALO=halos['XHALO']
    YHALO=halos['YHALO']
    ZHALO=halos['ZHALO']
    IHALO=halos['IHALOX']
    JHALO=halos['IHALOY']
    KHALO=halos['IHALOZ']

    skew_xy = Table([XHALO,YHALO,ZHALO,IHALO,JHALO,KHALO],names=('XHALO','YHALO','ZHALO','IHALO','JHALO','KHALO'),masked=True)

    skewers=vstack([skew_xy])
    skewers.write(skewerfile,overwrite=True)
    return True

# X, Y should be in Mpc, and not Mpc/h, LBOX in Mpc.
def lya_skewers_halos(X,Y,Z,Nskew,R_MIN,R_MAX,LBOX,Ng,rand,Ng_chunk):
    
    #xdb.set_trace()

    LOGRMIN = math.log10(R_MIN)
    LOGRMAX = math.log10(R_MAX)
    #LOGRMIN_BG = math.log10(RSMALL_MAX)
    #LOGRMAX_BG = math.log10(RBIG_MAX)

    nhalo=len(X)

    XSKEW=np.zeros(Nskew)
    YSKEW=np.zeros(Nskew)
    BEGIN=np.zeros(Nskew)
    END=np.zeros(Nskew)
    KHALO=np.zeros(Nskew)
    JHALO=np.zeros(Nskew)
    IHALO=np.zeros(Nskew)
    ZHALO=np.zeros(Nskew)
    YHALO=np.zeros(Nskew)
    XHALO=np.zeros(Nskew)
    if (Nskew>nhalo):
        phi_all=np.zeros(Nskew)

    # Draw a vector of small separations around each halo
    for ii in np.arange(Nskew):
        LOGR = LOGRMIN + rand.rand()*(LOGRMAX - LOGRMIN)
        R=np.power(10.0,LOGR)
        # In case there are halos with more than one skewer, the following
        # routine makes sure that skewers are not drawn too close to each other
        if (Nskew>nhalo):
            phi=draw_phi(ii,Nskew,nhalo,phi_all,rand)
            phi_all[ii]=phi
        else:
            phi = rand.rand()*np.pi*2.0
        DX = R*math.cos(phi)
        DY = R*math.sin(phi)
        XSKEW[ii]= X[ii%nhalo] + DX
        YSKEW[ii]= Y[ii%nhalo] + DY
        ZHALO[ii], KHALO[ii]=get_halo_coords(ii,Z,nhalo,LBOX,Ng)
        YHALO[ii], JHALO[ii]=get_halo_coords(ii,Y,nhalo,LBOX,Ng)
        XHALO[ii], IHALO[ii]=get_halo_coords(ii,X,nhalo,LBOX,Ng)
        BEGIN[ii]=(KHALO[ii]-Ng_chunk/2)%Ng
        END[ii]= (KHALO[ii]+Ng_chunk/2)%Ng
    
    # Clip the X Y values to always reside in the box. We are not using periodicity here
    XSKEW=np.clip(XSKEW,0.0,LBOX)
    YSKEW=np.clip(YSKEW,0.0,LBOX)

    # Convert to integer values on the grid
    IX=np.mod(np.trunc(XSKEW/LBOX*Ng).astype(int),Ng)
    IY=np.mod(np.trunc(YSKEW/LBOX*Ng).astype(int),Ng)
    return XSKEW, YSKEW,IX,IY,BEGIN,END,XHALO,YHALO,ZHALO,IHALO,JHALO,KHALO

def draw_phi(ii,Nskew,nhalo,phi_all,rand):
    phi=rand.rand()*np.pi*2.0
    if (ii>nhalo-1):
        sph = Nskew/nhalo # skewers per halo
        #print("skewers per halo=%s"%(sph))
        if (sph<2):
             sph=2
             #print("sph set to %s."%(sph))
        dphi_min=2.0*np.pi/(sph*4) # sets minimum angular separation between two halos
        #print("dphi_min=%s"%(dphi_min))
        aux=np.where(np.mod(np.arange(len(phi_all)),nhalo)==ii%nhalo)[0]
        phi_halo=phi_all[aux][0:len(aux)]
        while (np.any(np.minimum(np.fabs(phi-phi_halo),np.fabs(2.0*np.pi-phi+phi_halo))<dphi_min)):
            phi=rand.rand()*np.pi*2.0
        #print("%s done"%(ii))
    return phi


def get_halo_coords(ii,Z,nhalo,LBOX,Ng):
    ZHALO=Z[ii%nhalo]
    KHALO=math.trunc(ZHALO/LBOX*Ng)
    return ZHALO, KHALO

# X, Y should be in Mpc, and not Mpc/h, LBOX in Mpc. 
def lya_skewers_halos_old(X,Y,Z,nsmall,nbig,RSMALL_MIN,RSMALL_MAX,RBIG_MAX,LBOX,Ng,rand,Ng_chunk):
     
     #xdb.set_trace()

    LOGRMIN_SM = math.log10(RSMALL_MIN)
    LOGRMAX_SM = math.log10(RSMALL_MAX)
    LOGRMIN_BG = math.log10(RSMALL_MAX)
    LOGRMAX_BG = math.log10(RBIG_MAX)
    
    nhalo=len(X)
    
    XSMALL=np.zeros(nsmall*nhalo)
    YSMALL=np.zeros(nsmall*nhalo)
    BEGIN_SMALL=np.zeros(nsmall*nhalo)
    END_SMALL=np.zeros(nsmall*nhalo)
    KHALO_SMALL=np.zeros(nsmall*nhalo)
    # Draw a vector of small separations around each halo
    for ii in range(nsmall):
        LOGRSMALL = LOGRMIN_SM + rand.rand(nhalo)*(LOGRMAX_SM - LOGRMIN_SM)
        RSMALL=np.power(10.0,LOGRSMALL)
        phi = rand.rand(nhalo)*np.pi*2.0
        DXSMALL = RSMALL*np.cos(phi)
        DYSMALL = RSMALL*np.sin(phi)
        XSMALL[ii*nhalo:(ii+1)*nhalo]= X + DXSMALL
        YSMALL[ii*nhalo:(ii+1)*nhalo]= Y + DYSMALL
        HALO_POS=np.trunc(Z/LBOX*Ng).astype(int)
        KHALO_SMALL[ii*nhalo:(ii+1)*nhalo]=HALO_POS
        BEGIN_SMALL[ii*nhalo:(ii+1)*nhalo]= np.mod(HALO_POS-(Ng_chunk/2)*np.ones(len(Z)),Ng)
        END_SMALL[ii*nhalo:(ii+1)*nhalo]= np.mod(HALO_POS+(Ng_chunk/2)*np.ones(len(Z)),Ng)

    XBIG=np.zeros(nbig*nhalo)
    YBIG=np.zeros(nbig*nhalo)
    BEGIN_BIG=np.zeros(nbig*nhalo)
    END_BIG=np.zeros(nbig*nhalo)
    KHALO_BIG=np.zeros(nbig*nhalo)
    # Draw a vector of big separations around each halo
    for ii in range(nbig):
        LOGRBIG = LOGRMIN_BG + rand.rand(nhalo)*(LOGRMAX_BG - LOGRMIN_BG)
        RBIG=np.power(10.0,LOGRBIG)
        phi = rand.rand(nhalo)*np.pi*2.0
        DXBIG = RBIG*np.cos(phi)
        DYBIG = RBIG*np.sin(phi)
        XBIG[ii*nhalo:(ii+1)*nhalo]= X + DXBIG
        YBIG[ii*nhalo:(ii+1)*nhalo]= Y + DYBIG
        HALO_POS=np.trunc(Z/LBOX*Ng).astype(int)
        KHALO_BIG[ii*nhalo:(ii+1)*nhalo]=HALO_POS
        BEGIN_BIG[ii*nhalo:(ii+1)*nhalo]= np.mod(HALO_POS-(Ng_chunk/2)*np.ones(len(Z)),Ng)
        END_BIG[ii*nhalo:(ii+1)*nhalo]= np.mod(HALO_POS+(Ng_chunk/2)*np.ones(len(Z)),Ng)

    # Concatenate all the skewer locations into one array
    # Halo itself is *NOT* used as a skewer location
    XSKEW=np.concatenate((XSMALL,XBIG))
    YSKEW=np.concatenate((YSMALL,YBIG))
    BEGIN=np.concatenate((BEGIN_SMALL,BEGIN_BIG))
    END=np.concatenate((END_SMALL,END_BIG))
    KHALO=np.concatenate((KHALO_SMALL,KHALO_BIG))
    # Clip the X Y values to always reside in the box. We are not using periodicity here
    XSKEW=np.clip(XSKEW,0.0,LBOX)
    YSKEW=np.clip(YSKEW,0.0,LBOX)
   
    # Convert to integer values on the grid
    IX=np.mod(np.trunc(XSKEW/LBOX*Ng).astype(int),Ng)
    IY=np.mod(np.trunc(YSKEW/LBOX*Ng).astype(int),Ng)
    return XSKEW, YSKEW,IX,IY,BEGIN,END,KHALO

def lya_skewers_halo_positions(skewerfile,halofile,Nskew,LBOX,Ng):
    
    positions=ascii.read(halofile)

    #for i in np.arange(Nskew):
    # coordinates in the grid
    IX=positions['col1'][0:Nskew]
    IY=positions['col2'][0:Nskew]
    IZ=np.zeros(Nskew,dtype=int)
    
    # coordinates in the box (units Mpc)
    X=(IX*LBOX/Ng).astype(float)
    Y=(IY*LBOX/Ng).astype(float)
    Z=np.zeros(Nskew)
    
    # projection axis (x,y,z), i.e. z =2, y=1, x = 0)
    PROJ_AXIS=np.zeros(Nskew,dtype=int) + 2
    # direction of projection i.e. toward positive or negative values
    PROJ_SIGN=np.ones(Nskew,dtype=int)
    
    
    skewers = Table([X,Y,Z,IX,IY,IZ,PROJ_AXIS,PROJ_SIGN],names=('XSKEW','YSKEW','ZSKEW','ISKEWX','ISKEWY','ISKEWZ','PROJ_AXIS','PROJ_SIGN'),masked=True)
    # Mask the z-projection directions
    skewers['ISKEWZ'].mask[:]=True
    skewers['ZSKEW'].mask[:]=True
    skewers.write(skewerfile,overwrite=True)
        
        
    return True


def lya_skewers_random(skewerfile,Nskew,LBOX,Ng,rand):
    # coordinates in the grid
    IX=rand.randint(0,Ng,Nskew)
    IY=rand.randint(0,Ng,Nskew)
    IZ=np.zeros(Nskew,dtype=int)

    # coordinates in the box (units Mpc)
    X=(IX*LBOX/Ng).astype(float)
    Y=(IY*LBOX/Ng).astype(float)
    Z=np.zeros(Nskew)

    # projection axis (x,y,z), i.e. z =2, y=1, x = 0)
    PROJ_AXIS=np.zeros(Nskew,dtype=int) + 2
    # direction of projection i.e. toward positive or negative values
    PROJ_SIGN=np.ones(Nskew,dtype=int)


    skewers = Table([X,Y,Z,IX,IY,IZ,PROJ_AXIS,PROJ_SIGN],names=('XSKEW','YSKEW','ZSKEW','ISKEWX','ISKEWY','ISKEWZ','PROJ_AXIS','PROJ_SIGN'),masked=True)
    # Mask the z-projection directions
    skewers['ISKEWZ'].mask[:]=True
    skewers['ZSKEW'].mask[:]=True
    skewers.write(skewerfile,overwrite=True)

    return True

def lya_skewers_manual(skewerfile,spac,LBOX,Ng,blk,nblk):
    # coordinates in the grid
    #ax = np.round(np.linspace(0,Ng-1,LBOX/spac)).astype('int') #equally spaced numbers along a side
    ax = np.round((np.linspace(spac/2, LBOX-spac/2, LBOX/spac) - LBOX/Ng/2) / (LBOX/Ng))
    cofm = np.array(np.meshgrid(ax,ax,np.array([0.]))).T.reshape(-1,3) #all combinations of x and y
    cofm = np.array_split(cofm,nblk)[blk] #split up to preserve memory
    
    IX=cofm[:,0] #amend
    IY=cofm[:,1] #amend
    IZ=np.zeros(IX.shape,dtype=int) #amend
    
    Nskew = len(IX) #amend
    # coordinates in the box (units Mpc)
    X=(IX*LBOX/Ng).astype(float) #amend
    Y=(IY*LBOX/Ng).astype(float) #amend
    Z=np.zeros(Nskew) #amend
    
    # projection axis (x,y,z), i.e. z =2, y=1, x = 0)
    PROJ_AXIS=np.zeros(Nskew,dtype=int) + 2 #amend
    # direction of projection i.e. toward positive or negative values
    PROJ_SIGN=np.ones(Nskew,dtype=int)
    
    skewers = Table([X,Y,Z,IX,IY,IZ,PROJ_AXIS,PROJ_SIGN],names=('XSKEW','YSKEW','ZSKEW','ISKEWX','ISKEWY','ISKEWZ','PROJ_AXIS','PROJ_SIGN'),masked=True)
    # Mask the z-projection directions
    skewers['ISKEWZ'].mask[:]=True #amend
    skewers['ZSKEW'].mask[:]=True
    skewers.write(skewerfile,overwrite=True)

    return skewers

def lya_skewers_los(halos,Lbox,skewerfile,XPROJ=False,NEG_PROJ=False):

    # direction of projection i.e. toward positive or negative values
    nhalo=len(halos)
    PROJ_SIGN=np.ones(nhalo,dtype=int)

    # Project along z

    # projection axis (x,y,z), i.e. z =2, y=1, x = 0) 
    PROJ_AXIS=np.zeros(nhalo,dtype=int) + 2

    # coordinates in the grid
    IX=halos['IHALOX']
    IY=halos['IHALOY']
    IZ=0*IX

    # coordinates in the box (units Mpc)
    X=halos['XHALO']
    Y=halos['YHALO']
    Z=0*X

    R_LOS=halos['ZHALO']
    S_LOS=halos['SZHALO']

    skew_xy1 = Table([X,Y,Z,IX,IY,IZ,R_LOS,S_LOS,PROJ_AXIS,PROJ_SIGN]
                 ,names=('XSKEW','YSKEW','ZSKEW','ISKEWX','ISKEWY','ISKEWZ'
                         ,'R_LOS','S_LOS','PROJ_AXIS','PROJ_SIGN'),masked=True)
    skew_xy1['ISKEWZ'].mask[:]=True
    skew_xy1['ZSKEW'].mask[:]=True

    skew_xy=hstack([halos,skew_xy1],join_type='exact')

    if (NEG_PROJ==True):
        skew_xy_flip=skew_xy
        skewers_xy_flip['PROJ_SIGN']=-1
        skew_xy_flip['XHALO']=Lbox - skew_xy_flip['XHALO']
        skew_xy_flip['SXHALO']=Lbox - skew_xy_flip['SXHALO']
        skew_xy_flip['R_LOS']=Lbox - skew_xy_flip['R_LOS']
        skew_xy_flip['S_LOS']=Lbox - skew_xy_flip['S_LOS']
        skew_xy_flip['VXHALO']=-skew_xy_flip['VXHALO']
        skew_xy=vstack([skew_xy,skew_xy_flip])
          
        # Project along y

        # projection axis (x,y,z), i.e. z =2, y=1, x = 0) 
        PROJ_AXIS=np.zeros(nhalo,dtype=int) + 1

        # coordinates in the grid
        IX=halos['IHALOX']
        IY=0*IX
        IZ=halos['IHALOZ']

        # coordinates in the box (units Mpc)
        X=halos['XHALO']
        Y=0*X
        Z=halos['ZHALO']

        R_LOS=halos['YHALO']
        S_LOS=halos['SYHALO']

        skew_zx1 = Table([X,Y,Z,IX,IY,IZ,R_LOS,S_LOS,PROJ_AXIS,PROJ_SIGN],
                     names=('XSKEW','YSKEW','ZSKEW','ISKEWX','ISKEWY','ISKEWZ',
                            'R_LOS','S_LOS','PROJ_AXIS','PROJ_SIGN'),masked=True)
        skew_zx1['ISKEWY'].mask[:]=True
        skew_zx1['YSKEW'].mask[:]=True

        skew_zx=hstack([halos,skew_zx1],join_type='exact')

        if (NEG_PROJ==True):
            skew_zx_flip=skew_zx
            skewers_zx_flip['PROJ_SIGN']=-1
            skew_zx_flip['YHALO']=Lbox - skew_zx_flip['YHALO']
            skew_zx_flip['SYHALO']=Lbox - skew_zx_flip['SYHALO']
            skew_zx_flip['R_LOS']=Lbox - skew_zx_flip['R_LOS']
            skew_zx_flip['S_LOS']=Lbox - skew_zx_flip['S_LOS']
            skew_zx_flip['VYHALO']=-skew_zx_flip['VYHALO']
            skew_zx=vstack([skew_zx,skew_zx_flip])
     
     
    if (XPROJ==True):
        # Project along x

        # projection axis (x,y,z), i.e. z =2, y=1, x = 0) 
        PROJ_AXIS=np.zeros(nhalo,dtype=int)

        # coordinates in the grid
        IY=halos['IHALOY']
        IZ=halos['IHALOZ']
        IX=0*IZ

        # coordinates in the box (units Mpc)
        Y=halos['XHALO']
        Z=halos['ZHALO']
        X=0*Z

        R_LOS=halos['XHALO']
        S_LOS=halos['SXHALO']

        skew_yz1 = Table([X,Y,Z,IX,IY,IZ,R_LOS,S_LOS,PROJ_AXIS,PROJ_SIGN],
                      names=('XSKEW','YSKEW','ZSKEW','ISKEWX','ISKEWY','ISKEWZ',
                             'R_LOS','S_LOS','PROJ_AXIS','PROJ_SIGN'),masked=True)
        skew_yz1['ISKEWX'].mask[:]=True
        skew_yz1['XSKEW'].mask[:]=True

        skew_yz=hstack([halos,skew_yz1],join_type='exact')

        if (NEG_PROJ==True):
            skew_yz_flip=skew_yz
            skewers_yz_flip['PROJ_SIGN']=-1
            skew_yz_flip['XHALO']=Lbox - skew_yz_flip['XHALO']
            skew_yz_flip['SXHALO']=Lbox - skew_yz_flip['SXHALO']
            skew_yz_flip['R_LOS']=Lbox - skew_yz_flip['R_LOS']
            skew_yz_flip['S_LOS']=Lbox - skew_yz_flip['S_LOS']
            skew_yz_flip['VXHALO']=-skew_yz_flip['VXHALO']
            skew_yz=vstack([skew_yz,skew_yz_flip])
               
            skewers=vstack([skew_xy,skew_zx,skew_yz])
    else:
        skewers=vstack([skew_xy,skew_zx])



    skewers.write(skewerfile,overwrite=True)    


          
    return True



