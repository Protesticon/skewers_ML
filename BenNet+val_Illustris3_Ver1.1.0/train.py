import time
import numpy as np
import torch

from model import AverageMeter
from data_loader import make_batch_grids

'''
Training process. For missed parameters plz check the main.py.
'''

def train(train_ske, train_block, DM_general, DM_param,
        batch_size, train_size, model, criterion, optimizer,
        num_epochs, epoch, device, start_time, localtime):
    '''
    Global variables: len_ske, DM_param, start_time, date
    Possible: batch_size, train_size, num_epochs, device
    '''
    losses = AverageMeter()
    # switch to train mode
    model.train()

    for i, train_data in enumerate(train_ske, 0):
        # get the targets;
        targets = train_data.reshape((batch_size, 1)).to(device)
        # x,y,z are the central coordinates of each input DM cuboid
        x, y, z = train_block[(i*batch_size+np.arange(batch_size)).astype('int')].transpose()
        # make coordinate index, retrieve input dark matter
        batch_grids = make_batch_grids(x, y, z, batch_size, train_size, DM_param)
        inputs = DM_general[batch_grids].to(device)

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # record loss
        losses.update(loss.item(), batch_size)

        # zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}, Time: {:.4f}"
                   .format(epoch+1, num_epochs, i+1,
                           train_ske.shape[0], loss.item(), time.time()-start_time))
        if (i+1) % 10000 == 0: 
            print ("SAVING MODEL!")
            torch.save(model.state_dict(),
                       "params/HyPhy_%s"%time.strftime("%Y-%m-%d_%H:%M:%S", localtime))
    
    return losses.avg


'''
template training function
def train(train_loader, model, criterion, optimizer, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # Create non_blocking tensors for distributed training
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

        # compute gradients in a backward pass
        optimizer.zero_grad()
        loss.backward()

        # Call step of optimizer to update model params
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

def adjust_learning_rate(initial_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
'''