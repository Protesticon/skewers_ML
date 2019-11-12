from pathlib import Path
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import warnings
warnings.filterwarnings('ignore')

from data_loader import *
from model import *


def test(test_ske, test_block, DM_general, DM_param,
        test_batch, train_size, model, criterion,
         device, start_time):

    losses = AverageMeter()
    
    test_outp  = np.zeros(test_ske.shape).flatten()

    with torch.no_grad():
        for i, test_data in enumerate(test_ske, 0):

            # get the targets;
            targets = test_data.reshape((test_batch, 1)).to(device)
            # x,y,z are the central coordinates of each input DM cube
            x, y, z = test_block[(i*test_batch+np.arange(test_batch)).astype('int')].transpose()
            # make coordinate index, retrieve input dark matter
            batch_grids = make_batch_grids(x, y, z, test_batch, train_size, DM_param)
            inputs = DM_general[batch_grids].to(device)
            
            
            # compute output and mearsure/record loss
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.update(loss.item(), test_batch)
            
            # record outputs
            test_outp[i*test_batch:(i+1)*test_batch] = outputs.detach().cpu().numpy().flatten()

            if (i+1) % 100 == 0:
                print ("Step [{}/{}] Loss: {:.4f}, Time: {:.4f}"
                    .format(i+1, test_ske.shape[0], loss.item(), time.time()-start_time))
    
    test_outp = test_outp.reshape(-1, DM_param.pix)

    return test_outp, losses.avg



def test_plot(test_block_i, test_outp_i, test_ske_i, test_DM_i,
             vaxis, folder_outp):
    
    # plotting 1dps of skewers
    # k axis
    bins = int(15)
    len_ske = len(test_outp_i)
    rvax_t = np.arange(int(len_ske/2))
    logrv  = np.log10(rvax_t)
    logrv[0] = logrv[1]-10
    rvmin, rvmax = 0, logrv[-1]
    bin_sz = (rvmax-rvmin)/bins
    bins_l = np.arange(rvmin, rvmax, bin_sz).reshape(-1,1)
    bins_r = np.arange(rvmin+bin_sz, rvmax+bin_sz, bin_sz).reshape(-1,1)
    bin_bl = (logrv>=bins_l) & (logrv<bins_r)
    bin_bl[-1,-1] = True
    rvaxis = np.zeros(bins)
    # 1dps and pdf
    outp4fft = (test_outp_i-test_outp_i.mean())/test_outp_i.std()
    test4fft = (test_ske_i-test_ske_i.mean())/test_ske_i.std()
    fft_outp = np.absolute(np.fft.fft(outp4fft))[:int(len_ske/2)]
    fft_test = np.absolute(np.fft.fft(test4fft))[:int(len_ske/2)]
    onePS_outp = np.zeros(bins)
    onePS_test = np.zeros(bins)
    for ii in range(bins):
        rvaxis[ii] = 10**logrv[bin_bl[ii]].mean()
        onePS_outp[ii]  = (rvax_t*fft_outp**2)[bin_bl[ii]].mean() * 2
        onePS_test[ii]  = (rvax_t*fft_test**2)[bin_bl[ii]].mean() * 2
    rvaxis = rvaxis * 0.5/vaxis[1]/(len_ske/2)
    onePS_outp = onePS_outp * 0.5/vaxis[1]/(len_ske/2)
    onePS_test = onePS_test * 0.5/vaxis[1]/(len_ske/2)
    
    accuracy_i = (np.abs(onePS_outp[:-1]-onePS_test[:-1])/onePS_test[:-1]).mean()
    rela_err_i = (np.abs(onePS_outp[:-1]-onePS_test[:-1])/onePS_test[:-1]).std()
    
    ffig, axes = plt.subplots(1, 2, figsize=(12, 5))
    p1, = axes[0].plot(rvaxis, onePS_outp, label='Predicted')
    p2, = axes[0].plot(rvaxis, onePS_test, label='Real', alpha=0.5)
    axes[0].set_xlabel(r'$k\ (\mathrm{s/km})$', fontsize=18)
    axes[0].set_ylabel(r'$kP_\mathrm{1D}/\pi$', fontsize=18)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    customs = [p1, p2, 
              Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='k', markersize=5),
              Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='k', markersize=5)]

    axes[1].hist(test_outp_i, bins=np.arange(0,1,0.05),
              density=True, histtype='step', label='Predicted')
    axes[1].hist(test_ske_i, bins=np.arange(0,1,0.05),
              density=True, histtype='step', label='Real', alpha=0.5)
    axes[1].set_xlabel(r'normalized F', fontsize=18)
    axes[1].set_ylabel(r'pdf', fontsize=18)
    axes[1].set_xlim([-0.05, 1.05])
    axes[1].legend(fontsize=18, bbox_to_anchor=(1.06,0.75))
    axes[1].legend(customs, [p1.get_label(), p2.get_label(), '$m=%.3f$'%accuracy_i,
                        '$s=%.3f$'%rela_err_i], fontsize=18, bbox_to_anchor=(1.06,0.75))
    plt.savefig(folder_outp / \
        ('x%dy%d_1DPS&PDF.png'%(test_block_i[0], test_block_i[1])),
        dpi=200, bbox_inches='tight')
    plt.subplots_adjust(wspace=0.15, hspace=0)
    plt.close()
    
    

    # plotting skewers
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    p1, = axes[0].plot(vaxis, -np.log(test_outp_i), label='Predicted' )
    p2, = axes[0].plot(vaxis, -np.log(test_ske_i), label='Real', alpha=0.5)
    axes[0].set_xlabel(r'v (km/s)', fontsize=18)
    axes[0].set_ylabel(r'$\tau$', fontsize=18)
    axes[0].set_ylim([0,3])
    subaxs = axes[0].twinx()
    p3, = subaxs.plot(vaxis, test_DM_i,
                       label='DM', alpha=0.3, color='green' )
    subaxs.set_ylim([0, 5])
    subaxs.set_ylabel(r'DM Over Den+1', fontsize=18)
    axes[0].legend([p1,p2,p3], [l.get_label() for l in [p1,p2,p3]],
                   fontsize=18, bbox_to_anchor=(1.26,0.75))

    p1, = axes[1].plot(vaxis, test_outp_i, label='Predicted', alpha=0.7 )
    p2, = axes[1].plot(vaxis, test_ske_i, label='Real', alpha=0.5 )
    axes[1].set_xlabel(r'v (km/s)', fontsize=18)
    axes[1].set_ylabel(r'$F = \mathrm{e}^{-\tau}$', fontsize=18)
    axes[1].set_ylim([-0.1, 1.1])
    customs = [p1, p2, 
              Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='k', markersize=5),
              Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='k', markersize=5)]
    axes[1].legend(customs, [p1.get_label(), p2.get_label(), '%.3f'%accuracy_i,
                            '%.3f'%rela_err_i], fontsize=18, bbox_to_anchor=(1.26,0.75))

    axes[1].get_shared_x_axes().join(axes[1], axes[0])
    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.savefig(folder_outp / \
        ('x%dy%d_skewer.png'%(test_block_i[0], test_block_i[1])),
        dpi=200, bbox_inches='tight')
    plt.close()
    
    return accuracy_i, rela_err_i



def test_accuracy(test_block_i, test_outp_i, test_ske_i,
                 vaxis, folder_outp):
    bins = int(15)
    len_ske = len(test_outp_i)
    rvax_t = np.arange(int(len_ske/2))
    logrv  = np.log10(rvax_t)
    logrv[0] = logrv[1]-10
    rvmin, rvmax = 0, logrv[-1]
    bin_sz = (rvmax-rvmin)/bins
    bins_l = np.arange(rvmin, rvmax, bin_sz).reshape(-1,1)
    bins_r = np.arange(rvmin+bin_sz, rvmax+bin_sz, bin_sz).reshape(-1,1)
    bin_bl = (logrv>=bins_l) & (logrv<bins_r)
    bin_bl[-1,-1] = True
    rvaxis = np.zeros(bins)
    # 1DPS
    outp4fft = (test_outp_i-test_outp_i.mean())/test_outp_i.std()
    test4fft = (test_ske_i-test_ske_i.mean())/test_ske_i.std()
    fft_outp = np.absolute(np.fft.fft(outp4fft))[:int(len_ske/2)]
    fft_test = np.absolute(np.fft.fft(test4fft))[:int(len_ske/2)]
    onePS_outp = np.zeros(bins)
    onePS_test = np.zeros(bins)
    for ii in range(bins):
        rvaxis[ii] = 10**logrv[bin_bl[ii]].mean()
        onePS_outp[ii]  = (rvax_t*fft_outp**2)[bin_bl[ii]].mean() * 2
        onePS_test[ii]  = (rvax_t*fft_test**2)[bin_bl[ii]].mean() * 2
    rvaxis = rvaxis * 0.5/vaxis[1]/(len_ske/2)
    onePS_outp = onePS_outp * 0.5/vaxis[1]/(len_ske/2)
    onePS_test = onePS_test * 0.5/vaxis[1]/(len_ske/2)
    
    accuracy_i = (np.abs(onePS_outp[:-1]-onePS_test[:-1])/onePS_test[:-1]).mean()
    rela_err_i = (np.abs(onePS_outp[:-1]-onePS_test[:-1])/onePS_test[:-1]).std()
    
    return accuracy_i, rela_err_i