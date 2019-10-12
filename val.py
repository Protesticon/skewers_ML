import time
import numpy as np
import torch

from model import AverageMeter, accuracy
from data_loader import make_batch_grids

'''
Validation process. For missed parameters plz check the main.py.
'''


def validate(val_ske, val_block, ske_len, DM_general, DM_param,
        batch_size, train_size, model, criterion, device, start_time):

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, val_data in enumerate(val_ske, 0):

            # get the targets;
            targets = val_data.reshape((batch_size,1)).to(device)
            # x,y,z are the central coordinates of each training DM cube
            x = (val_block[np.floor((i*batch_size+np.arange(batch_size))/ske_len).astype('int'), 0]-DM_param.reso/2)/DM_param.reso
            y = (val_block[np.floor((i*batch_size+np.arange(batch_size))/ske_len).astype('int'), 1]-DM_param.reso/2)/DM_param.reso
            z = np.linspace(start=0, stop=ske_len-1, num=ske_len)[(i*batch_size+np.arange(batch_size))%ske_len] # z from 0 or to 0?
            # make coordinate index, retrieve input dark matter
            batch_grids = make_batch_grids(x, y, z, batch_size, train_size, DM_general.shape[1])
            inputs = DM_general[batch_grids].to(device)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))


            if (i+1) % 100 == 0:
                print (" Step [{}/{}] Loss: {:.4f},Time: {:.4f}"
                    .format(i+1, val_ske.shape[0], loss.item(), time.time()-start_time))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg