from pathlib import Path
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from data_loader import *
from model import *


def test(test_ske, test_block, ske_len, DM_general, DM_param,
        test_batch, train_size, model, criterion,
         device, start_time):

    losses = AverageMeter()
    
    test_outp  = np.zeros(test_ske.shape).flatten()

    with torch.no_grad():
        for i, test_data in enumerate(test_ske, 0):

            # get the targets;
            targets = test_data.reshape((test_batch,1)).to(device)
            # x,y,z are the central coordinates of each training DM cube
            x = (test_block[np.floor((i*test_batch+np.arange(test_batch))/ske_len).astype('int'), 0]-DM_param.reso/2)/DM_param.reso
            y = (test_block[np.floor((i*test_batch+np.arange(test_batch))/ske_len).astype('int'), 1]-DM_param.reso/2)/DM_param.reso
            z = np.linspace(start=0, stop=ske_len-1, num=ske_len)[(i*test_batch+np.arange(test_batch))%ske_len] # z from 0 or to 0?
            # make coordinate index, retrieve input dark matter
            batch_grids = make_batch_grids(x, y, z, test_batch, train_size, DM_general.shape[1])
            inputs = DM_general[batch_grids].to(device)

            # compute output
            outputs = model(inputs).detach().cpu().numpy().flatten()
            test_outp[i*test_batch:(i+1)*test_batch] = outputs

            # measure and record loss
            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0)) 

            if (i+1) % 100 == 0:
                print (" Step [{}/{}] Loss: {:.4f}, Time: {:.4f}"
                    .format(i+1, test_ske.shape[0], loss.item(), time.time()-start_time))
    
    test_outp = test_outp.reshape(-1, DM_param.pix)

    return test_outp, losses.avg


# Path and data file name
folder  = Path.cwd().parent / 'Illustris3'
DM_name = ['DMdelta_Illustris3_L75_N600.fits', 
            'vx_cic_Illustris3_L75_N600.fits',
            'vy_cic_Illustris3_L75_N600.fits',
            'vz_cic_Illustris3_L75_N600.fits']
ske_name = 'spectra_Illustris3_N600.dat'



# hyper parameters
train_size = np.array([9, 9, 17]) # x, y, z respctively
test_batch = 40
learning_rate = 0.0001
num_epochs = 10
localtime  = ???
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
ske, block = load_skewers(folder, ske_name, DM_param.reso)
# pre-procession
ske = 1-np.exp(-1*ske)
## only use the skewers that lie on DM pixels
boolean = ((block[:,0]+DM_param.reso/2)%DM_param.reso + (block[:,1]+DM_param.reso/2)%DM_param.reso) == 0
ske     = ske[boolean]
block   = block[boolean]
del boolean
## only use skewers that satisfy certain requirments
# basic parameters
ske_len = ske.shape[1]


# divide the sample to training, validation set, and test set.
print('Setting test set...')
with open("id_seperate/id_seperate_%s.txt"\
        %time.strftime("%Y-%m-%d_%H:%M:%S", localtime), "r") as f:
    aa = f.readlines()
    id_seperate = np.array(list(aa[0][1:-1][::3])).astype('int')
    del aa
f.close()

test_ske, test_block = load_test(ske, block, id_seperate, test_batch)
test_ske = torch.FloatTensor(test_ske)
del id_seperate


# load model
model = get_residual_network().float().to(device)
model.load_state_dict(torch.load('params/params_%s.pkl'\
        %time.strftime("%Y-%m-%d_%H:%M:%S", localtime)))
# model.load_state_dict(torch.load('params/HyPhy_.pkl'))


criterion = nn.MSELoss()

start_time = time.time()

test_outp, test_losses = test(test_ske, test_block, ske_len, DM_general, DM_param,
                        test_batch, train_size, model, criterion, device, start_time)

test_ske   = test_ske.numpy().reshape(-1, DM_param.pix)
test_block = ((test_block-DM_reso/2)/DM_reso).astype('int')

if not os.path.exists(Path.cwd() / 'test_figs' / ('%s'\
        %time.strftime("%Y-%m-%d_%H:%M:%S", localtime))):
    os.makedirs(Path.cwd() / 'test_figs' / ('%s'\
        %time.strftime("%Y-%m-%d_%H:%M:%S", localtime)))

for ii, outputs in enumerate(test_outp):
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot( np.exp(test_outp[ii]), label='Prediction' )
    axes[0].plot( np.exp(test_ske[ii]), label='Real', alpha=0.5 )
    #axes[0].plot( DM_general[0, int(test_block[ii,0]/DM_reso), int(test_block[ii,1]/DM_reso), :].numpy(), label='DM', alpha=0.3 )
    axes[0].set_ylabel(r'$\tau$', fontsize=18)
    axes[0].set_ylim([0,3])
    axes[0].legend(fontsize=18, bbox_to_anchor=(1.26,0.75))

    axes[1].plot( np.exp(-np.exp(test_outp[ii])), label='Prediction', alpha=0.7 )
    axes[1].plot( np.exp(-np.exp(test_ske[ii])), label='Real', alpha=0.5 )
    axes[1].set_ylabel(r'$F = \mathrm{e}^{-\tau}$', fontsize=18)
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].legend(fontsize=18, bbox_to_anchor=(1.26,0.75))

    axes[1].get_shared_x_axes().join(axes[1], axes[0])
    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.savefig(Path.cwd() / 'test_figs' / ('%s'\
        %time.strftime("%Y-%m-%d_%H:%M:%S", localtime)) / \
        ('x%dy%d.png'%(test_block[ii,0], test_block[ii,1])),
        dpi=400, bbox_inches='tight')