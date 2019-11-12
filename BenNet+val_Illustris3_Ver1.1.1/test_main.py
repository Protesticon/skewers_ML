from pathlib import Path
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from data_loader import *
from test import *



# pre-process
def pre_proc(tau, block):
    '''1-exp(-tau)'''
    tau   = np.array(tau)
    block = np.array(block)
    return (1-np.exp(-1*tau), block)

def toF_proc(tau):
    '''transfer data derived from pre_proc to F=exp(-tau)'''
    tau   = np.array(tau)
    return (1-tau)



# Path and data file name
folder  = Path.cwd().parent / 'Illustris3'
DM_name = ['DMdelta_Illustris3_L75_N600_v2.fits', 
            'vx_cic_Illustris3_L75_N600.fits',
            'vy_cic_Illustris3_L75_N600.fits',
            'vz_cic_Illustris3_L75_N600.fits']
ske_name = 'spectra_Illustris3_N600.npy'



# hyper parameters
train_size = np.array([9, 9, 67]) # x, y, z respctively
test_batch = 50
localtime_n = ['2019-11-03 03:45:29', '2019-11-04 14:30:32']
for localtime_i in localtime_n:
    localtime = time.strptime(localtime_i, '%Y-%m-%d %H:%M:%S')
    if ~(train_size%2).all():
        raise ValueError('train size scannot be even.')



    # device used to train the model
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)



    # load dark matter data
    print('Loading dark matter...')
    DM_general = load_DM(folder, DM_name)
    # basic paramters
    DM_param.pix  = len(DM_general[0])
    DM_param.len  = 75*1000 # in kpc/h
    DM_param.reso = DM_param.len / DM_param.pix # in kpc/h
    # test
    if DM_general.shape[1]<train_size.min():
        raise ValueError('DarkMatter cube size',
            DM_general.shape, 'is too small for train size', train_size, '.')
    DM_general = torch.tensor(DM_general).float()


    # load skewers
    print('Loading skewers...')
    ske, block = load_skewers(folder, ske_name, DM_param)
    # basic parameters
    ske_len = ske.shape[1]


    # divide the sample to training, validation set, and test set.
    print('Setting test set...')
    with open("id_seperate/id_seperate_%s.txt"\
              %time.strftime("%Y-%m-%d_%H:%M:%S", localtime), "r") as f:
        aa = f.readlines()
        id_seperate = np.array(list(aa[0][::3])).astype('int')
        del aa
    f.close()

    test_ske, test_block = load_test(ske, block, id_seperate, test_batch)
    test_ske, test_block = pre_proc(test_ske, test_block)
    test_ske = torch.FloatTensor(test_ske)
    del id_seperate


    # load model
    model = get_residual_network().float().to(device)
    model.load_state_dict(torch.load('params/params_%s.pkl'\
            %time.strftime("%Y-%m-%d_%H:%M:%S", localtime)))
    # model.load_state_dict(torch.load('params/HyPhy_%s.pkl'\
    #       %time.strftime("%Y-%m-%d_%H:%M:%S", localtime)))



    # loss
    criterion = nn.SmoothL1Loss()


    # record starr time
    start_time = time.time()


    # start test
    print('Begin testing...')
    test_outp, test_losses = test(test_ske, test_block, DM_general, DM_param,
                            test_batch, train_size, model, criterion, device, start_time)

    print("Test Summary: ")
    print("\tTest loss: {}".format(test_losses))

    # restore test skewers
    test_ske   = test_ske.numpy().reshape(-1, ske_len)
    test_ske   = toF_proc(test_ske)
    test_outp  = toF_proc(test_outp)
    test_block = test_block.reshape(-1, DM_param.pix, 3)

    
    print('Plotting example skewers...')
    # generate comparison images
    folder_outp = Path.cwd()/'test_figs'/('%s'\
            %time.strftime("%Y-%m-%d_%H:%M:%S", localtime))
    if not os.path.exists(folder_outp):
        os.makedirs(folder_outp)

    from scipy import constants as C
    v_end  = 0.02514741843009228 * C.speed_of_light / 1e3
    vaxis  = np.arange(0, v_end, v_end/600)
    
    nrange = min(len(test_ske), 50)
    test_sp = np.arange(len(test_ske))
    np.random.seed(99)
    np.random.shuffle(test_sp)
    test_sp1 = test_sp[:int(nrange)].astype('int')
    test_sp2 = test_sp[int(nrange):].astype('int')
    
    accuracy = AverageMeter()
    rela_err = AverageMeter()
    
    #loop
    for i, ii in enumerate(test_sp1):
        print('Plotting {}/{}, x{}y{}.png...'\
              .format((i+1), nrange, test_block[ii,0,0], test_block[ii,0,1]))
        
        test_block_i = test_block[ii, 0]
        test_outp_i = test_outp[ii]
        test_ske_i = test_ske[ii]
        test_DM_i = DM_general[0, test_block[ii,0,0], test_block[ii,0,1], :].numpy()
        
        accuracy_i, rela_err_i = test_plot(test_block_i, test_outp_i, test_ske_i, test_DM_i,
                 vaxis, folder_outp)
        
        accuracy.update(accuracy_i, 1)
        rela_err.update(rela_err_i, 1)
        
    print('Measuring accuracy of left skewers...')
    for i, ii in enumerate(test_sp2):
        
        test_block_i = test_block[ii, 0]
        test_outp_i = test_outp[ii]
        test_ske_i = test_ske[ii]
        test_DM_i = DM_general[0, test_block[ii,0,0], test_block[ii,0,1], :].numpy()
        
        accuracy_i, rela_err_i = test_accuracy(test_block_i, test_outp_i,
                                               test_ske_i, vaxis, folder_outp)
        accuracy.update(accuracy_i, 1)
        rela_err.update(rela_err_i, 1)
        
        

    # record this test
    with open('history.txt', 'a') as f:
        f.writelines('\n\n\nTest History Record:')
        f.writelines('\n\tTest of the training at %s'\
                %time.strftime("%Y-%m-%d %H:%M:%S", localtime))
        f.writelines('\n\tTest loss: %s,  '%str(test_losses)\
            +time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime()))
        f.writelines('\n\tAverage accuracy: %s,  '%str(accuracy.avg)\
            +time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime()))
        f.writelines('\n\tAverage relative error: %s,  '%str(rela_err.avg)\
            +time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime()))
    f.close()
    
print('Finished test!')