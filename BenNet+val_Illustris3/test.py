from pathlib import Path
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

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
            x = test_block[(i*test_batch+np.arange(test_batch)).astype('int'), 0]
            y = test_block[(i*test_batch+np.arange(test_batch)).astype('int'), 1]
            z = test_block[(i*test_batch+np.arange(test_batch)).astype('int'), 2]
            # make coordinate index, retrieve input dark matter
            batch_grids = make_batch_grids(x, y, z, test_batch, train_size, DM_param)
            inputs = DM_general[batch_grids].to(device)
            
            
            # compute output and mearsure/record loss
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            
            # record outputs
            test_outp[i*test_batch:(i+1)*test_batch] = outputs.detach().cpu().numpy().flatten()

            if (i+1) % 100 == 0:
                print ("Step [{}/{}] Loss: {:.4f}, Time: {:.4f}"
                    .format(i+1, test_ske.shape[0], loss.item(), time.time()-start_time))
    
    test_outp = test_outp.reshape(-1, DM_param.pix)

    return test_outp, losses.avg



# pre-process
def pre_proc(tau, block):
    '''1-exp(tau)'''
    return (1-np.exp(-1*tau), block)



# Path and data file name
folder  = Path.cwd().parent / 'Illustris3'
DM_name = ['DMdelta_Illustris3_L75_N600.fits', 
            'vx_cic_Illustris3_L75_N600.fits',
            'vy_cic_Illustris3_L75_N600.fits',
            'vz_cic_Illustris3_L75_N600.fits']
ske_name = 'spectra_Illustris3_N600.dat'



# hyper parameters
train_size = np.array([9, 9, 17]) # x, y, z respctively
test_batch = 50
learning_rate = 0.0001
num_epochs = 10
localtime = '2019-10-15 13:21:22'
localtime = time.strptime(localtime, '%Y-%m-%d %H:%M:%S')
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

# restore test skewers and test blocks
test_ske   = test_ske.numpy().reshape(-1, DM_param.pix)
test_block = ((test_block-DM_param.reso/2)/DM_param.reso).astype('int')


# generate comparison images
if not os.path.exists(Path.cwd() / 'test_figs' / ('%s'\
        %time.strftime("%Y-%m-%d_%H:%M:%S", localtime))):
    os.makedirs(Path.cwd() / 'test_figs' / ('%s'\
        %time.strftime("%Y-%m-%d_%H:%M:%S", localtime)))

print('Plotting example skewers...')
nrange = min(len(test_ske), 50)
for ii in range(nrange):
    print('Plotting {}/{}...'.format((ii+1), nrange))
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot( -np.log(1-test_outp[ii]), label='Predicted' )
    axes[0].plot( -np.log(1-test_ske[ii]), label='Real', alpha=0.5 )
    #axes[0].plot( DM_general[0, int(test_block[ii,0]/DM_reso), int(test_block[ii,1]/DM_reso), :].numpy(), label='DM', alpha=0.3 )
    axes[0].set_ylabel(r'$\tau$', fontsize=18)
    axes[0].set_ylim([0,3])
    axes[0].legend(fontsize=18, bbox_to_anchor=(1.26,0.75))

    axes[1].plot( 1-test_outp[ii], label='Predicted', alpha=0.7 )
    axes[1].plot( 1-test_ske[ii], label='Real', alpha=0.5 )
    axes[1].set_ylabel(r'$F = \mathrm{e}^{-\tau}$', fontsize=18)
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].legend(fontsize=18, bbox_to_anchor=(1.26,0.75))

    axes[1].get_shared_x_axes().join(axes[1], axes[0])
    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.savefig(Path.cwd() / 'test_figs' / ('%s'\
        %time.strftime("%Y-%m-%d_%H:%M:%S", localtime)) / \
        ('x%dy%d.png'%(test_block[ii,0], test_block[ii,1])),
        dpi=400, bbox_inches='tight')
    plt.close()

# record this test
with open('history.txt', 'a') as f:
    f.writelines('\n\n\nTest History Record:')
    f.writelines('\n\tTest of the training at %s.'\
            %time.strftime("%Y-%m-%d, %H:%M:%S", localtime))
    f.writelines('\n\tTest loss: %s,  '%str(test_losses)\
        +time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime()))
f.close()