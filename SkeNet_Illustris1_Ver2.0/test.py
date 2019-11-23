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
        test_batch, train_size, model, mdn_loss,
         device, start_time):

    losses = AverageMeter()
    
    # switch to eval mode
    model.eval()
    
    test_outp = np.zeros(test_ske.shape[0], test_ske.shape[1])

    with torch.no_grad():
        for i, test_data in enumerate(test_ske, 0):

            # get the targets;
            targets = test_data.to(device)
            # x,y,z are the central coordinates of each input DM cube
            x, y, z = test_block[(i*test_batch+np.arange(test_batch)).astype('int')].T
            # make coordinate index, retrieve input dark matter
            batch_grids = make_batch_grids(x, y, z, test_batch, train_size, DM_param)
            inputs = DM_general[batch_grids].to(device)
            
            
            # compute output and mearsure/record loss
            pi, sigma, mu = model(inputs)
            loss = mdn_loss(pi, sigma, mu, targets)
            losses.update(loss.item(), test_batch)
            test_outp = sample(pi, sigma, mu).squeeze()
            
            # record outputs
            test_outp[i] = outputs.detach().cpu().numpy()

            if (i+1) % 100 == 0:
                print("Step [{:{}d}/{}] Loss: {:.4f}, Time: {:.4f}"
                    .format(i+1, int(np.log10(test_ske.shape[0])+1),
                            test_ske.shape[0], loss.item(), time.time()-start_time))

    return test_outp, losses.avg



def test_plot(test_block_i, test_outp_i, test_ske_i,
        test_DM_i, F_mean, v_end, folder_outp, bins):
    
    # plotting 1dps and pdf of skewers
    # k axis
    ske_len = len(test_outp_i)
    vaxis  = np.arange(0, v_end, v_end/ske_len)
    rvax_t = np.arange(int(ske_len/2))
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
    outp4fft = (test_outp_i-F_mean[1])/F_mean[1]
    test4fft = (test_ske_i -F_mean[0])/F_mean[0]
    fft_outp = np.absolute(np.fft.fft(outp4fft))[:int(ske_len/2)]
    fft_test = np.absolute(np.fft.fft(test4fft))[:int(ske_len/2)]
    onePS_outp = np.zeros(bins)
    onePS_test = np.zeros(bins)
    for jj in range(bins):
        rvaxis[jj] = 10**logrv[bin_bl[jj]].mean() 
        onePS_outp[jj]  = (rvax_t*fft_outp**2)[bin_bl[jj]].mean() * 2
        onePS_test[jj]  = (rvax_t*fft_test**2)[bin_bl[jj]].mean() * 2
    rvaxis = rvaxis * 2 * np.pi / v_end
    onePS_outp = onePS_outp * 2 * np.pi / v_end
    onePS_test = onePS_test * 2 * np.pi / v_end
    
    accuracy_i = (np.abs(onePS_outp-onePS_test)/onePS_test).mean()
    rela_err_i = (np.abs(onePS_outp-onePS_test)/onePS_test).std()
    
    
    fig = plt.figure(figsize=(12,8))
    # plot skewers in F
    axes1 = fig.add_subplot(2,1,1)
    p1, = axes1.plot(vaxis, test_outp_i, label='Predicted', alpha=0.7 )
    p2, = axes1.plot(vaxis, test_ske_i, label='Real', alpha=0.5 )
    axes1.set_xlabel(r'$v$ (km/s)', fontsize=18, labelpad=0)
    axes1.set_ylabel(r'$F = \mathrm{e}^{-\tau}$', fontsize=18)
    axes1.set_ylim([-0.1, 1.1])
    axes1.tick_params(labelsize=12, direction='in')

    # plot 1DPS
    axes2 = fig.add_subplot(2,2,3)
    axes2.plot(rvaxis, onePS_outp, label='Predicted')
    axes2.plot(rvaxis, onePS_test, label='Real', alpha=0.5)
    axes2.set_xlabel(r'$k\ (\mathrm{s/km})$', fontsize=18)
    axes2.set_ylabel(r'$kP_\mathrm{1D}/\pi$', fontsize=18)
    axes2.set_xscale('log')
    axes2.set_yscale('log')
    axes2.tick_params(labelsize=12, direction='in')

    # plot pdf of F
    axes3 = fig.add_subplot(2,2,4)
    p3 = axes3.hist(test_outp_i, bins=np.arange(0,1.05,0.05),
              density=True, histtype='step', label='Predicted')
    p4 = axes3.hist(test_ske_i, bins=np.arange(0,1.05,0.05),
              density=True, histtype='step', label='Real', alpha=0.5)
    axes3.set_xlabel(r'$F$', fontsize=18)
    axes3.set_ylabel(r'pdf', fontsize=18)
    axes3.set_xlim([-0.05, 1.05])
    axes3.tick_params(labelsize=12, direction='in')
    customs = [p1, p2, 
              Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='k', markersize=5),
              Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='k', markersize=5)]
    axes3.legend(customs, [p1.get_label(), p2.get_label(), '$m=%.3f$'%accuracy_i,
                        '$s=%.3f$'%rela_err_i], fontsize=18, bbox_to_anchor=(1.06,1.5))
    plt.subplots_adjust(wspace=0.18, hspace=0.23)
    plt.savefig(folder_outp / \
        ('x%dy%d.png'%(test_block_i[0], test_block_i[1])),
        dpi=200, bbox_inches='tight')
    plt.close()
    
    stat_i = np.array([rvaxis, onePS_outp, onePS_test, accuracy_i, rela_err_i])
    
    return stat_i



def test_accuracy(test_block_i, test_outp_i, test_ske_i,
        F_mean, v_end, folder_outp, bins):
    ske_len = len(test_outp_i)
    vaxis  = np.arange(0, v_end, v_end/ske_len)
    rvax_t = np.arange(int(ske_len/2))
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
    fft_outp = np.absolute(np.fft.fft(outp4fft))[:int(ske_len/2)]
    fft_test = np.absolute(np.fft.fft(test4fft))[:int(ske_len/2)]
    onePS_outp = np.zeros(bins)
    onePS_test = np.zeros(bins)
    for jj in range(bins):
        rvaxis[jj] = 10**logrv[bin_bl[jj]].mean()
        onePS_outp[jj]  = (rvax_t*fft_outp**2)[bin_bl[jj]].mean() * 2
        onePS_test[jj]  = (rvax_t*fft_test**2)[bin_bl[jj]].mean() * 2
    rvaxis = rvaxis * 2 * np.pi / v_end
    onePS_outp = onePS_outp * 2 * np.pi / v_end
    onePS_test = onePS_test * 2 * np.pi / v_end
    
    accuracy_i = (np.abs(onePS_outp-onePS_test)/onePS_test).mean()
    rela_err_i = (np.abs(onePS_outp-onePS_test)/onePS_test).std()
    
    stat_i = np.array([rvaxis, onePS_outp, onePS_test, accuracy_i, rela_err_i])
    
    return stat_i