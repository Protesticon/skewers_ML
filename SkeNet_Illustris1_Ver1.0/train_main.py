from pathlib import Path
import time
import numpy as np
import torch

from data_loader import *
from model import *
from train import *
from val import *

# Path and data file name
folder  = Path.cwd().parent / 'Illustris1'
DM_name = ['deltaDM_Illustris1_L75_N600_v2.fits', 
            'vx_DM_Illustris1_L75_N600.fits',
            'vy_DM_Illustris1_L75_N600.fits',
            'vz_DM_Illustris1_L75_N600.fits']
ske_name_tra = 'spectra_Illustris1_N600_zaxis.npy'
ske_name_val = 'spectra_Illustris1_N600_yaxis.npy'


# hyper parameters
train_len  = 90000 # number of tau blocks
val_len    = 9000  # number of tau blocks
test_len   = 9000  # number of skewers
train_insize = np.array([11, 11, 141]) # x, y, z respctively
train_ousize = np.array([1, 1, 75]) # x, y, z respctively
batch_size = 50
learning_rate = 0.0005
weight_decay = 0.05
factor = 0.1
patience = 2
num_epochs = 20
localtime = time.localtime()
if ~(train_insize%2).all():
    raise ValueError('train size cannot be even.')

# pre-process
def pre_proc(tau, block):
    '''log(tau), 97%'''
    F_sum = np.exp(-tau).sum(axis=(-1,-2,-3))
    limit = np.percentile(F_sum, 10)
    bln = F_sum >= limit#np.ones(len(tau), 'bool')
    tau = np.log(tau)
    return (tau[bln],  block[bln])



# device used to train the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', torch.cuda.get_device_name(device=device.index))



# load dark matter data
print('Loading dark matter...')
DM_general_tra = load_DM(folder, DM_name)
DM_general_tra = DM_general_tra#.transpose(0,2,3,1)[[0,2,3,1]]
DM_general_val = load_DM(folder, DM_name)
DM_general_val = DM_general_val.transpose(0,3,1,2)[[0,3,1,2]]
# basic paramters
DM_param.pix  = len(DM_general_tra[0])
DM_param.len  = 75 # in Mpc/h
DM_param.reso = DM_param.len / DM_param.pix # in Mpc/h
# test 
if DM_general_tra.shape[1]<train_insize.min():
    raise ValueError('DarkMatter cube size',
        DM_general_tra.shape, 'is too small for train size', train_insize, '.')
DM_general_tra = torch.tensor(DM_general_tra).float()
DM_general_val = torch.tensor(DM_general_val).float()


# load skewers
print('Loading skewers...')
ske_tra, block_tra = load_skewers(folder, ske_name_tra, train_ousize, DM_param)
ske_val, block_val = load_skewers(folder, ske_name_val, train_ousize, DM_param)
# basic parameters
ske_len = int(ske_tra.shape[-1])

from scipy import optimize
def norm_specs(skewers,redshift):
    tauevo = 0.001845*(1+redshift)**3.924 #optical depth corresponding to the mean-flux of Faucher-Giguere et al. 2008 
    fobs = np.exp(-tauevo)
    def fun(a): return np.mean(np.exp(-a*skewers)-fobs)
    a = optimize.newton(fun, 1)    
    return a
rs = 2.0020281392528516 #redshift of the snapshot

tau = ske_tra
normalization = norm_specs(tau,rs) #determine the normalization constant on the whole tau box
tau_normed = normalization*tau #normalized optical depth
ske_tra = normalization*tau #normalized flux

tau = ske_val
normalization = norm_specs(tau,rs) #determine the normalization constant on the whole tau box
tau_normed = normalization*tau #normalized optical depth
ske_val = normalization*tau #normalized flux


# divide the sample to training, validation set, and test set.
print('Setting training and validation set...')
id_seperate = divide_data(ske_tra, train_ousize, train_len, val_len, test_len, localtime)

train_ske, train_block = load_train(ske_tra, block_tra, id_seperate,
                                    train_ousize, batch_size, pre_proc)

val_ske, val_block = load_val(ske_val, block_val, id_seperate,
                              train_ousize, batch_size, pre_proc)

del id_seperate


print('Loading model, loss, optimizer etc...')
# load model
model = get_residual_network().float().to(device)
# loss and optimizer
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=2)


# Train the model
curr_lr = learning_rate
start_time = time.time()
val_time   = localtime

lowest_losses = 9999.0
lowest_time   = localtime
tra_loss_l = np.zeros(num_epochs)
val_loss_l = np.zeros(num_epochs)

print('\nStart Training:')
with open('history.txt', 'a') as f:
    f.writelines('\n\n\nTraining History Record,')
    f.writelines('\nTime: '+time.strftime("%Y-%m-%d %H:%M:%S", localtime))
    f.writelines('\nTrain Frac: {}/{}'.format(len(train_ske.flatten()), len(ske_tra.flatten())))
    f.writelines('\nReal Train Frac: {}/{}'.format(len(train_ske.flatten()), len(ske_tra.flatten())))
    f.writelines('\nVal Frac: {}/{}'.format(len(val_ske.flatten()), len(ske_val.flatten())))
    f.writelines('\nReal Val Frac: {}/{}'.format(len(val_ske.flatten()), len(ske_val.flatten())))
    f.writelines('\nInput Size: %s'%str(train_insize))
    f.writelines('\nOutput Size: %s'%str(train_ousize))
    f.writelines('\nTraining Transformation: %s'%(pre_proc.__doc__))
    f.writelines('\nInput DM Transformation: log(delta)')
    f.writelines('\nLoss: %s'%criterion.__class__.__name__)
    f.writelines('\nOptimizer: %s'%optimizer.__class__.__name__)
    f.writelines('\nLearning Rate: %s'%str(learning_rate))
    f.writelines('\nWeight Decay Rate: %s'%str(weight_decay))
    f.writelines('\nFactor: %s'%str(factor))
    f.writelines('\nPatience: %s'%str(patience))
f.close()


for epoch in range(num_epochs):
    # train for one epoch
    print("Begin Training Epoch {}".format(epoch+1))
    train_losses = train(train_ske, train_block, DM_general_tra, DM_param,
                    batch_size, train_insize, model, criterion, optimizer,
                    num_epochs, epoch, device, start_time, localtime)
    
    with open('history.txt', 'a') as f:
        f.writelines('\nEpoch {:{}d}/{}:'.format(epoch+1,
                                                 int(np.log10(num_epochs)+1), num_epochs))
        f.writelines('\n\tTraining loss : %s,  '%str(train_losses)\
            +time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime()))
    f.close()

    # evaluate on validation set
    print("Begin Validation @ Epoch {}".format(epoch+1))
    val_losses = validate(val_ske, val_block, DM_general_val, DM_param,
                batch_size, train_insize, model, criterion, device, start_time)
    val_time   = time.localtime()
    
    
    # remember best prec@1 and save checkpoint if desired
    # is_best = prec1 > best_prec1
    torch.save(model.state_dict(), "params/params_%s_%d.pkl"%(time.strftime("%Y-%m-%d_%H:%M:%S", localtime), epoch+1))
    if train_losses < lowest_losses:
        lowest_losses = train_losses
        lowest_time   = val_time
        torch.save(model.state_dict(),
             "params/params_%s.pkl"%time.strftime("%Y-%m-%d_%H:%M:%S", localtime))
    
    else: model.load_state_dict(torch.load('params/params_%s.pkl'\
                %time.strftime("%Y-%m-%d_%H:%M:%S", localtime)))
    
    # reducing learning rate if val_losses do not reduce
    scheduler.step(val_losses)
    
    with open('history.txt', 'a') as f:
        f.writelines('\n\tValidation loss: %s,  '%str(val_losses)\
            +time.strftime("%Y-%m-%d, %H:%M:%S", val_time))
        f.writelines('\n\tLowest val loss: %s,  '%str(lowest_losses)\
            +time.strftime("%Y-%m-%d, %H:%M:%S", lowest_time))
    f.close()

    print("Epoch Summary: ")
    print("\tEpoch training loss: {}".format(train_losses))
    print("\tEpoch validation loss: {}".format(val_losses))
    print("\tLowest training loss: {}".format(lowest_losses))    




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