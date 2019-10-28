import numpy as np
from astropy.io import fits
from more_itertools import chunked


def make_batch_grids(x, y, z, batch_size, train_size, DM_param):
    '''
    To get the coordinate (index) of dark matter to be retrieved given the central pixel's coordinate and the input size.

    x, y, z: int, coordinate of the central pixel

    batch_size: int, the number of trained samples at one time;

    train_size: 3d array of int in order of x,y,z, the size of each trained sample;
    '''
    # x,y,z range are coordinates ranges of each training cube
    x_range = ((x.reshape(batch_size,-1)+(np.arange(train_size[0])-(train_size[0]-1)/2))%DM_param.pix).astype('int')
    y_range = ((y.reshape(batch_size,-1)+(np.arange(train_size[1])-(train_size[1]-1)/2))%DM_param.pix).astype('int')
    z_range = ((z.reshape(batch_size,-1)+(np.arange(train_size[2])-(train_size[2]-1)/2))%DM_param.pix).astype('int')

    # cx,cy,cz are coordinates of every points of each training cube, together forming a meshgrid
    ci = np.array([0,1,2,3]).repeat(train_size.prod()*batch_size).reshape(4,batch_size,train_size[0],train_size[1],train_size[2]).transpose(1,0,2,3,4)
    cx = x_range.repeat(train_size[[1,2]].prod()).reshape(batch_size,1,train_size[1],train_size[0],train_size[2]).transpose(0,1,2,3,4).repeat(4, axis=1)
    cy = y_range.repeat(train_size[[2,0]].prod()).reshape(batch_size,1,train_size[0],train_size[2],train_size[1]).transpose(0,1,4,2,3).repeat(4, axis=1)
    cz = z_range.repeat(train_size[[0,1]].prod()).reshape(batch_size,1,train_size[2],train_size[0],train_size[1]).transpose(0,1,3,4,2).repeat(4, axis=1)
    
    return tuple([ci, cx, cy, cz])


class DM_param(object):
    def __init__(self):
        self.reso # in Mpc/h
        self.len  # in kpc/h
        self.pix  # in pixels


def load_DM(Path, FileName):
    '''
    To load the dark matter data. Output is in the form of [over density + 1, normalized v_x, normalized v_y, normalized v_z]. Velocities are normalized by being divided by the average of the absolute value of each field. This is to make 4 fields (or channels) have the similar digit.
    '''
    # read in over density field
    DM_fits = fits.open(Path/FileName[0])
    DM = DM_fits[0].data + 1
    DM_fits.close(); del DM_fits

    # basic paramters
    DM_pix  = len(DM)
    DM_len  = 75*1000 # in kpc/h
    DM_reso = DM_len / DM_pix # in kpc/h

    # read in vx field
    DM_vx_fits = fits.open(Path/FileName[1])
    DM_vx = DM_vx_fits[0].data
    DM_vx_fits.close(); del DM_vx_fits

    # read in vy field
    DM_vy_fits = fits.open(Path/FileName[2])
    DM_vy = DM_vy_fits[0].data
    DM_vy_fits.close(); del DM_vy_fits

    # read in vz field
    DM_vz_fits = fits.open(Path/FileName[3])
    DM_vz = DM_vz_fits[0].data
    DM_vz_fits.close(); del DM_vz_fits

    # normalize the velocity field
    v_mean = np.linalg.norm(([DM_vx, DM_vy, DM_vz]), axis=0).mean()

    # put 4 fields into 1 numpy array
    DM_general = np.array([DM, DM_vx/v_mean, DM_vy/v_mean, DM_vz/v_mean])
    del DM, DM_vx, DM_vy, DM_vz, v_mean
    
    return DM_general



def load_skewers(Path, FileName, DM_param):
    '''
    To load original skewers data in shape of [number, length in pixels]. Generating each coordinate [x, y, 0] simultaneously. Output: skewers and coordinate.
    '''
    # read in skewers and make coordinates of the skewers
    ske   = np.loadtxt(Path/FileName)
    ax = np.arange(DM_param.pix)
    block = np.array(np.meshgrid(ax, ax, ax)).transpose(2,1,3,0).reshape(-1,DM_param.pix,3)
    del ax

    return ske, block



def divide_data(ske, train_len, val_len, test_len, localtime):
    '''
    randomly selet the training set, validation set, and test set.
    '''
    import time
    waste_len = ske.shape[0] - train_len - val_len - test_len
    train_arr = np.ones(train_len)
    val_arr   = np.ones(val_len)*2
    test_arr  = np.ones(test_len)*3
    waste_arr = np.zeros(waste_len)
    id_seperate = np.concatenate((train_arr, val_arr, test_arr, waste_arr), axis=0)
    np.random.shuffle( id_seperate )
    with open("id_seperate/id_seperate_%s.txt"\
              %time.strftime("%Y-%m-%d_%H:%M:%S", localtime), "w") as f:
        f.writelines(str(list(id_seperate.astype('int')))[1:-1])
    f.close()

    return id_seperate



def load_train(ske, block, id_seperate, batch_size, pre_proc):
    '''
    To load, shuffle and chunk the training set.
    '''
    train_block = block[id_seperate == 1]
    train_ske   = ske[id_seperate == 1]

    np.random.seed(np.random.randint(0,50))
    state = np.random.get_state()
    np.random.shuffle( train_block )
    np.random.set_state(state)
    np.random.shuffle( train_ske )
    
    train_ske   = train_ske.flatten()
    train_block = train_block.reshape(-1, 3)
    train_ske, train_block = pre_proc(train_ske, train_block)
    train_len1  = len(train_ske) - len(train_ske)%batch_size
    train_ske   = train_ske[:train_len1]
    train_block = train_block[:train_len1]
    train_ske = list(chunked(train_ske, batch_size))
    
    return (train_ske, train_block)



def load_val(ske, block, id_seperate, batch_size, pre_proc):
    '''
    To load, shuffle and chunk the validation set.
    '''
    val_block = block[id_seperate == 2]
    val_ske   = ske[id_seperate == 2]

    np.random.seed(np.random.randint(0,51))
    state = np.random.get_state()
    np.random.shuffle( val_block )
    np.random.set_state(state)
    np.random.shuffle( val_ske )

    val_ske = val_ske.flatten()
    val_block = val_block.reshape(-1, 3)
    val_ske, val_block = pre_proc(val_ske, val_block)
    val_len1  = len(val_ske) - len(val_ske)%batch_size
    val_ske   = val_ske[:val_len1]
    val_block = val_block[:val_len1]
    val_ske = list(chunked( val_ske, batch_size )) 

    return (val_ske, val_block)



def load_test(ske, block, id_seperate, batch_size):
    '''
    To load and shuffle the test set.
    '''
    test_block = block[id_seperate == 3]
    test_ske   = ske[id_seperate == 3]

    np.random.seed(52)
    state = np.random.get_state()
    np.random.shuffle( test_block )
    np.random.set_state(state)
    np.random.shuffle( test_ske )

    test_ske   = test_ske.flatten()
    test_block = test_block.reshape(-1, 3)
    test_ske = list(chunked( test_ske, batch_size )) 

    return (test_ske, test_block)
    

class ske_param(object):
    def __init__(self):
        self.len