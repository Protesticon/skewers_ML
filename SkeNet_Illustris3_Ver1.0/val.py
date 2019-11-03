import time
import numpy as np
import torch

from model import AverageMeter
from data_loader import make_batch_grids

'''
Validation process. For missed parameters plz check the main.py.
'''


def validate(val_ske, val_block, DM_general, DM_param,
        batch_size, train_insize, model, criterion, device, start_time):

    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, val_data in enumerate(val_ske, 0):

            # get the targets;
            targets = val_data.to(device)
            # x,y,z are the central coordinates of each training DM cube
            x, y, z = val_block[(i*batch_size+np.arange(batch_size)).astype('int')].transpose()
            # make coordinate index, retrieve input dark matter
            batch_grids = make_batch_grids(x, y, z, batch_size, train_insize, DM_param)
            inputs = DM_general[batch_grids].to(device)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            losses.update(loss.item(), batch_size)


            if (i+1) % 100 == 0:
                print ("Step [{}/{}] Loss: {:.4f}, Time: {:.4f}"
                    .format(i+1, val_ske.shape[0], loss.item(), time.time()-start_time))

    return losses.avg


'''
def validate(val_loader, model, criterion):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg
    '''