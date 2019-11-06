from pathlib import Path
import time
import numpy as np
import torch

from data_loader import *
from model import *
from train import *
from val import *

# Path and data file name
folder  = Path.cwd().parent / 'Illustris3'
DM_name = ['DMdelta_Illustris3_L75_N600_v2.fits', 
            'vx_cic_Illustris3_L75_N600.fits',
            'vy_cic_Illustris3_L75_N600.fits',
            'vz_cic_Illustris3_L75_N600.fits']
ske_name = 'spectra_Illustris3_N600.npy'



# hyper parameters
train_len  = 400000 # number of tau blocks
val_len    = 20000  # number of tau blocks
test_len   = 20000  # number of skewers
train_insize = np.array([15, 15, 71]) # x, y, z respctively
train_ousize = np.array([5, 5, 5]) # x, y, z respctively
batch_size = 40
learning_rate = 0.001
num_epochs = 10
localtime = time.localtime()
if ~(train_insize%2).all():
    raise ValueError('train size scannot be even.')

# pre-process
def pre_proc(tau, block):
    '''1-exp(-tau)'''
    bln = np.ones(len(block), dtype='bool')
    tau = 1 - np.exp(-1*tau)
    return (tau[bln],  block[bln])



# device used to train the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)



# load dark matter data
print('Loading dark matter...')
DM_general = load_DM(folder, DM_name)
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
ske, block = load_skewers(folder, ske_name, train_ousize, DM_param)
# basic parameters
ske_len = ske.shape[1]


# divide the sample to training, validation set, and test set.
print('Setting training and validation set...')
id_seperate = divide_data(ske, train_ousize, train_len, val_len, test_len, localtime)

train_ske, train_block = load_train(ske, block, id_seperate, batch_size, pre_proc)

val_ske, val_block = load_val(ske, block, id_seperate, batch_size, pre_proc)

del id_seperate


# load model
model = get_residual_network().float().to(device)
# loss and optimizer
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train the model
curr_lr = learning_rate
start_time = time.time()
val_time   = localtime

lowest_losses = 9999.0
lowest_time   = localtime

print('\nStart Training:')
with open('history.txt', 'a') as f:
    f.writelines('\n\n\nTraining History Record:')
    f.writelines('\nTime: '+time.strftime("%Y-%m-%d %H:%M:%S", localtime))
    f.writelines('\nTrain Frac: {}/{}'.format(train_len*train_ousize.prod(), len(ske.flatten())))
    f.writelines('\nReal Train Frac: {}/{}'.format(len(train_ske)*batch_size*train_ousize.prod(), len(ske.flatten())))
    f.writelines('\nVal Frac: {}/{}'.format(val_len*train_ousize.prod(), len(ske.flatten())))
    f.writelines('\nReal Val Frac: {}/{}'.format(len(val_ske)*batch_size*train_ousize.prod(), len(ske.flatten())))
    f.writelines('\nInput Size: %s'%str(train_insize))
    f.writelines('\nOutput Size: %s'%str(train_ousize))
    f.writelines('\nTraining Field: %s'%(pre_proc.__doc__))
    f.writelines('\nLoss: %s'%criterion.__class__.__name__)
    f.writelines('\nOptimizer: %s'%optimizer.__class__.__name__)
    f.writelines('\nLearning Rate: %s'%str(learning_rate))
f.close()

for epoch in range(num_epochs):
    # train for one epoch
    print("Begin Training Epoch {}".format(epoch+1))
    train_losses = train(train_ske, train_block, DM_general, DM_param,
                    batch_size, train_insize, model, criterion, optimizer,
                    num_epochs, epoch, device, start_time, localtime)
    with open('history.txt', 'a') as f:
        f.writelines('\nEpoch {}/{}:'.format(epoch+1, num_epochs))
        f.writelines('\n\tTraining loss : %s,  '%str(train_losses)\
            +time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime()))
    f.close()

    # evaluate on validation set
    print("Begin Validation @ Epoch {}".format(epoch+1))
    val_losses = validate(val_ske, val_block, DM_general, DM_param,
                batch_size, train_insize, model, criterion, device, start_time)
    val_time   = time.localtime()

    # remember best prec@1 and save checkpoint if desired
    # is_best = prec1 > best_prec1
    if val_losses < lowest_losses:
        lowest_losses = val_losses
        lowest_time   = val_time
        torch.save(model.state_dict(),
             "params/params_%s.pkl"%time.strftime("%Y-%m-%d_%H:%M:%S", localtime))
    
    with open('history.txt', 'a') as f:
        f.writelines('\n\tValidation loss: %s,  '%str(val_losses)\
            +time.strftime("%Y-%m-%d, %H:%M:%S", val_time))
        f.writelines('\n\tLowest val loss: %s,  '%str(lowest_losses)\
            +time.strftime("%Y-%m-%d, %H:%M:%S", lowest_time))
    f.close()

    print("Epoch Summary: ")
    print("\tEpoch training loss: {}".format(train_losses))
    print("\tEpoch validation loss: {}".format(val_losses))
    print("\tLowest validation loss: {}".format(lowest_losses))
    




'''
best_prec1 = 0

for epoch in range(num_epochs):
    # Set epoch count for DistributedSampler
    train_sampler.set_epoch(epoch)

    # Adjust learning rate according to schedule
    adjust_learning_rate(starting_lr, optimizer, epoch)

    # train for one epoch
    print("\nBegin Training Epoch {}".format(epoch+1))
    train(train_loader, model, criterion, optimizer, epoch)

    # evaluate on validation set
    print("Begin Validation @ Epoch {}".format(epoch+1))
    prec1 = validate(val_loader, model, criterion)

    # remember best prec@1 and save checkpoint if desired
    # is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)

    print("Epoch Summary: ")
    print("\tEpoch Accuracy: {}".format(prec1))
    print("\tBest Accuracy: {}".format(best_prec1))
'''