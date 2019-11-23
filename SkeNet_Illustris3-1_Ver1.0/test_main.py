from pathlib import Path
import os
import time
import numpy as np
import torch

import warnings
warnings.filterwarnings('ignore')

from data_loader import *
from model import *
from test import *


# pre-process
def pre_proc(tau, block):
    '''log(tau)'''
    tau   = np.log(tau)
    block = np.array(block)
    return (tau, block)

def toF_proc(tau):
    '''transfer data derived from pre_proc to F=exp(-tau)'''
    tau = np.exp(-np.exp(tau))
    return tau


# Path and data file name
folder3  = Path.cwd().parent / 'Illustris3'
folder1  = Path.cwd().parent / 'Illustris1'
DM_name = ['DMdelta_Illustris3_L75_N600_v2.fits', 
            'vx_cic_Illustris3_L75_N600_v2.fits',
            'vy_cic_Illustris3_L75_N600_v2.fits',
            'vz_cic_Illustris3_L75_N600_v2.fits']
ske_name = 'spectra_Illustris1_N600_zaxis.npy'



# hyper parameters
train_insize = np.array([15, 15, 71]) # x, y, z respctively
train_ousize = np.array([5, 5, 5]) # x, y, z respctively
test_batch = 50
localtime_n = ['2019-11-15 04:39:02']
for localtime_i in localtime_n:
    localtime = time.strptime(localtime_i, '%Y-%m-%d %H:%M:%S')

    
    
    # device used to train the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)



    # load dark matter data
    print('Loading dark matter...')
    DM_general = load_DM(folder3, DM_name)
    DM_general = DM_general#.transpose(0,2,3,1)
    #DM_general = DM_general[[0,2,3,1]]
    # basic paramters
    DM_param.pix  = len(DM_general[0])
    DM_param.len  = 75 # in Mpc/h
    DM_param.reso = DM_param.len / DM_param.pix # in Mpc/h
    # test
    if DM_general.shape[1]<train_insize.min():
        raise ValueError('DarkMatter cube size',
            DM_general.shape, 'is too small for train size', train_insize, '.')
    DM_general = torch.tensor(DM_general).float()


    # load skewers
    print('Loading skewers...')
    ske, block = load_skewers(folder1, ske_name, train_ousize, DM_param)
    # basic parameters
    ske_len = int(ske.shape[-1])


    # divide the sample to training, validation set, and test set.
    print('Setting test set...')
    with open("id_seperate/id_seperate_%s.txt"\
              %time.strftime("%Y-%m-%d_%H:%M:%S", localtime), "r") as f:
        aa = f.readlines()
        id_seperate = np.array(list(aa[0][::3])).astype('int')
        del aa
    f.close()

    test_ske, test_block = load_test(ske, block, id_seperate,
                                     train_ousize, test_batch, pre_proc)
    del id_seperate


    # load model
    print('Loading model...')
    model = get_residual_network().float().to(device)
    model.load_state_dict(torch.load('params/params_%s.pkl'\
            %time.strftime("%Y-%m-%d_%H:%M:%S", localtime)))
    #model.load_state_dict(torch.load('params/HyPhy_%s'\
    #        %time.strftime("%Y-%m-%d_%H:%M:%S", localtime)))



    # loss
    criterion = nn.MSELoss()


    # record starr time
    start_time = time.time()


    # start test
    print('Begin testing...')
    test_outp, test_losses = test(test_ske, test_block, DM_general, DM_param,
                            test_batch, train_insize, model, criterion, device, start_time)

    print("Test Summary: ")
    print("\tTest loss: {}".format(test_losses))
    
    # restore test skewers
    print('Restoring test skewers...')
    nz = (ske_len/train_ousize[2]).astype('int')
    test_outp = test_outp.reshape(-1, nz, train_ousize[0],
                                train_ousize[1], train_ousize[2])\
                                .transpose(0, 2, 3, 1, 4).reshape(-1, ske_len)
    test_ske = test_ske.numpy().reshape(-1, nz, train_ousize[0],
                                train_ousize[1], train_ousize[2])\
                                .transpose(0, 2, 3, 1, 4).reshape(-1, ske_len)
    test_outp = toF_proc(test_outp)
    test_ske  = toF_proc(test_ske)
    test_coord = test_block.reshape(-1, nz, 3)[:, 0, 0:2].T.reshape(2, 1, 1, -1)
    xcoor = np.arange(train_ousize[0]) - np.arange(train_ousize[0]).mean()
    ycoor = np.arange(train_ousize[1]) - np.arange(train_ousize[1]).mean()
    mesh  = np.expand_dims(np.array(np.meshgrid(xcoor, ycoor)), -1)
    test_block = (test_coord + mesh).transpose(3,2,1,0).reshape(-1, 2).astype(int)
    del test_coord, xcoor, ycoor, mesh


    print('Plotting example skewers...')
    # generate comparison images
    folder_outp = Path.cwd()/'test_figs'/('%s_x'\
            %time.strftime("%Y-%m-%d_%H:%M:%S", localtime))
    if not os.path.exists(folder_outp):
        os.makedirs(folder_outp)
    
    
    from scipy import constants as C
    v_end  = 0.02514741843009228 * C.speed_of_light / 1e3
    F_mean = np.array([test_ske.mean(), test_outp.mean()])
    
    nrange = min(len(test_ske), 50)
    test_sp = np.arange(len(test_ske))
    np.random.seed(99)
    np.random.shuffle(test_sp)
    test_sp1 = test_sp[:int(nrange)].astype('int')
    test_sp2 = test_sp[int(nrange):].astype('int')
    
    bins = int(15)
    accuracy = AverageMeter()
    rela_err = AverageMeter()
    accu_arr = np.zeros(len(test_ske))
    erro_arr = np.zeros(len(test_ske))
    oneDPS   = np.zeros(shape=(3, len(test_ske), bins))
    

    # loop
    for i, ii in enumerate(test_sp1):
        print('Plotting {:{}d}/{}, y{:03d}z{:03d}.png...'\
                .format((i+1), int(np.log10(nrange)+1), nrange,
                        test_block[ii,0], test_block[ii,1]))

        test_block_i = test_block[ii]
        test_outp_i = test_outp[ii]
        test_ske_i = test_ske[ii]
        test_DM_i = DM_general[0, test_block_i[0], test_block_i[1], :].numpy()

        stat_i = test_plot(test_block_i, test_outp_i, test_ske_i,
                          test_DM_i, F_mean, v_end, folder_outp, bins)
        accuracy_i, rela_err_i = stat_i[[3,4]]
        accuracy.update(accuracy_i, 1)
        rela_err.update(rela_err_i, 1)
        accu_arr[ii] = accuracy_i
        erro_arr[ii] = rela_err_i
        oneDPS[:,ii] = stat_i[0], stat_i[1], stat_i[2]
    
    print('Measuring accuracy of the left skewers...')
    for i, ii in enumerate(test_sp2):
        
        test_block_i = test_block[ii]
        test_outp_i = test_outp[ii]
        test_ske_i = test_ske[ii]
        test_DM_i = DM_general[0, test_block_i[0], test_block_i[1], :].numpy()
        
        stat_i = test_accuracy(test_block_i, test_outp_i, test_ske_i,
                              F_mean, v_end, folder_outp, bins)
        accuracy_i, rela_err_i = stat_i[[3,4]]
        accuracy.update(accuracy_i, 1)
        rela_err.update(rela_err_i, 1)
        accu_arr[ii] = accuracy_i
        erro_arr[ii] = rela_err_i
        oneDPS[:,ii] = stat_i[0], stat_i[1], stat_i[2]
    
    
    print('Plotting average 1DPS and histogram...')
    oneDPS = oneDPS.mean(axis=1)
    accuracy_gen = np.abs((oneDPS[1]-oneDPS[2])/oneDPS[2])[oneDPS[0]<0.1].mean()
    rela_err_gen = np.abs((oneDPS[1]-oneDPS[2])/oneDPS[2])[oneDPS[0]<0.1].std()
    
    outp_hist, F_hist = np.histogram(test_outp, bins=np.arange(0,1.05,0.05))
    outp_hist = np.append(outp_hist, outp_hist[-1]) / len(test_ske)
    test_hist, F_hist = np.histogram(test_ske, bins=np.arange(0,1.05,0.05))
    test_hist = np.append(test_hist, test_hist[-1]) / len(test_ske)
    accuracy_hist = np.abs((outp_hist[:-1]-test_hist[:-1])/test_hist[:-1]).mean()
    rela_err_hist = np.abs((outp_hist[:-1]-test_hist[:-1])/test_hist[:-1]).std()
    
    
    fig, axes = plt.subplots(2,2,figsize=(12,11))

    p0=axes[0,0].hist(accu_arr, color='grey', bins=np.arange(0, 1.7, 0.1))
    axes[0,0].set_ylim(axes[0,0].get_ylim())
    p1 = axes[0,0].vlines(x=accu_arr.mean(), ymin=0, ymax=9999, linestyle='--')
    axes[0,0].set_xlabel('accuracy $m$', fontsize=14)
    axes[0,0].set_title('pdf of $m$', fontsize=14)
    axes[0,0].tick_params(labelsize=12, direction='in')
    customs = [p1, 
              Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='k', markersize=5)]
    axes[0,0].legend(customs, ['average $m=%.4f$'%accu_arr.mean(),
                            '$N=%d$'%len(accu_arr)], fontsize=12, loc=1)

    axes[0,1].hist(erro_arr, color='grey', bins=np.arange(0, 1.7, 0.1))
    axes[0,1].set_ylim(axes[0,1].get_ylim())
    p2 = axes[0,1].vlines(x=erro_arr.mean(), ymin=0, ymax=9999, linestyle='--')
    axes[0,1].set_xlabel('error $s$', fontsize=14)
    axes[0,1].set_ylabel('pdf of $s$', fontsize=14)
    axes[0,1].set_title('pdf of $m$', fontsize=14)
    axes[0,1].tick_params(labelsize=12, direction='in')
    customs = [p2, 
              Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='k', markersize=5)]
    axes[0,1].legend(customs, ['average $s=%.4f$'%erro_arr.mean(),
                            '$N=%d$'%len(erro_arr)], fontsize=12, loc=1)

    p3, = axes[1,0].plot(oneDPS[0], oneDPS[1], label='Predicted')
    p4, = axes[1,0].plot(oneDPS[0], oneDPS[2], label='Real', alpha=0.5)
    axes[1,0].set_xlabel(r'$k\ (\mathrm{s/km})$', fontsize=14)
    axes[1,0].set_ylabel(r'$kP_\mathrm{1D}/\pi$', fontsize=14)
    axes[1,0].set_xscale('log')
    axes[1,0].set_yscale('log')
    axes[1,0].set_ylim(axes[0,1].get_ylim())
    axes[1,0].vlines(x=0.1, ymin=1e-8, ymax=1e8)
    axes[1,0].set_title('Average 1DPS', fontsize=14)
    axes[1,0].tick_params(labelsize=12, direction='in', which='both')
    customs = [p3, p4, 
              Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='k', markersize=5),
              Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='k', markersize=5)]
    axes[1,0].legend(customs, [p3.get_label(), p4.get_label(), '$m=%.3f$'%accuracy_gen,
                        '$s=%.3f$'%rela_err_gen], fontsize=12)

    p5, = axes[1,1].step(F_hist, outp_hist, where='post', label='Predicted')
    p6, = axes[1,1].step(F_hist, test_hist, where='post', label='Real', alpha=0.5)
    axes[1,1].set_xlabel(r'$F$', fontsize=18)
    axes[1,1].set_ylabel(r'Counts', fontsize=18)
    axes[1,1].set_xlim([-0.05, 1.05])
    axes[1,1].set_title('Average Histogram of $F$', fontsize=14)
    axes[1,1].tick_params(labelsize=12, direction='in')
    customs = [p5, p6, 
              Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='k', markersize=5),
              Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='k', markersize=5)]
    axes[1,1].legend(customs, [p5.get_label(), p6.get_label(), '$m=%.3f$'%accuracy_hist,
                        '$s=%.3f$'%rela_err_hist], fontsize=12)
    
    plt.savefig(folder_outp / ('average_S.png'), dpi=300, bbox_inches='tight') 
    plt.close()
    
    
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