import numpy as np
from pathlib import Path
from more_itertools import chunked
from datetime import date

from data_loader import *
from model import *
from train import *
from val import *

# Path and data file name
folder  = Path('../Illustris3/')
DM_name = ['DMdelta_Illustris3_L75_N600.fits', 
            'vx_cic_Illustris3_L75_N600.fits',
            'vy_cic_Illustris3_L75_N600.fits',
            'vz_cic_Illustris3_L75_N600.fits']
ske_name = 'spectra_Illustris3_N600.dat'



# hyper parameters
train_len  = 90000
val_len    = 900
test_len   = 400
train_size = np.array([9, 9, 17]) # x, y, z respctively
batch_size = 40
if ~(train_size%2).all():
    raise ValueError('train size scannot be even.')
learning_rate = 0.0001
num_epochs = 10



# device used to train the model
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print('Using device:', device)



# load dark matter data
DM_general = load_DM(folder, DM_name)
# basic paramters
DM_param.pix  = len(DM_general[0])
DM_param.len  = 75*1000 # in kpc/h
DM_param.reso = DM_param.len / DM_param.pix # in kpc/h
# test
if DM_general.shape[1]<train_size.min():
    raise ValueError('DarkMatter cube size',
        DM_general.shape, 'is too small for train size', train_size, '.')


# load skewers
ske, block = load_skewers(folder, ske_name, DM_param.reso)
# pre-procession
ske = 1-np.exp(-1*ske)
## only use the skewers that lie on DM pixels
boolean = ((block[:,0]+DM_param.reso/2)%DM_param.reso + (block[:,1]+DM_param.reso/2)%DM_param.reso) == 0
ske     = ske[boolean]
block   = block[boolean]
del boolean
print('There are %d skewers on DM pixels.'%ske.shape[0])
## only use skewers that satisfy certain requirments
# basic parameters
ske_len = ske.shape[1]


# divide the sample to training, validation set, and test set.
id_seperate = divide_data(ske, train_len, val_len, test_len)
train_ske, train_block = load_train(ske, block, id_seperate)
val_ske,   val_block   = load_val(ske, block, id_seperate)
with open("id_seperate.txt","w") as f:
    f.writelines(id_seperate.astype('str'))
f.close()
del id_seperate

# flatten the optical depth data and chunk in batches
train_ske = train_ske.flatten()
train_ske = torch.FloatTensor( list(chunked( train_ske, batch_size )) )


# load model
model = get_residual_network().float().to(device)
# loss and optimizer
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train the model
today = date.today()
curr_lr = learning_rate
start_time = time.time()

best_prec1 = 0

for epoch in range(num_epochs):
    # train for one epoch
    print("\nBegin Training Epoch {}".format(epoch+1))
    train(train_ske, train_block, ske_len, DM_general, DM_param,
        batch_size, train_size, model, criterion, optimizer,
        num_epochs, epoch, device, start_time, today)

    # evaluate on validation set
    print("Begin Validation @ Epoch {}".format(epoch+1))
    prec1 = validate(val_ske, val_block, ske_len, DM_general, DM_param,
        batch_size, train_size, model, criterion, device, start_time)

    # remember best prec@1 and save checkpoint if desired
    # is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)

    print("Epoch Summary: ")
    print("\tEpoch Accuracy: {}".format(prec1))
    print("\tBest Accuracy: {}".format(best_prec1))
