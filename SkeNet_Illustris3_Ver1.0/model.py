#imports for model
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

def narrow_tensor(tensorA, tensorB):
    '''
    to narrow tensorA to be like tensorB by deleting border elments 
    '''
    Ashape = np.array(tensorA.shape[2:5])
    Bshape = np.array(tensorB.shape[2:5])
    dx, dy, dz = ((Ashape-Bshape)//2)
    tensorA = tensorA[:,:,dx:Ashape[0]-dx,dy:Ashape[1]-dy,dz:Ashape[2]-dz]
    #print(tensorA.shape, tensorB.shape)
    return tensorA

#basic convolutional block
def conv3x3(in_channels, out_channels, stride=1, kernel_size=(3,3,3)):
    return nn.Conv3d(in_channels, out_channels, kernel_size, 
                     stride=stride, padding=0, bias=False)

#basic residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        
        residual = narrow_tensor(residual, out)
        out += residual
        
        return out
    

#The SkeNet!
def get_residual_network() -> torch.nn.Module:
    layers= [2, 2]
    class ResNet(torch.nn.Module):
        def __init__(self, block, layers = layers, num_classes=1):
            super(ResNet, self).__init__()
            self.in_channels = 16
            self.conv1 = conv3x3(4, 16)
            self.layer1 = self.make_layer(block, 16, layers[0])
            self.layer2 = self.make_layer(block, 32, layers[1], stride=1)
            self.conv2 = conv3x3(32, 8, kernel_size=(1,1,41))
            self.conv3 = conv3x3(8, 1)
            self.bn = torch.nn.BatchNorm3d(16)
            self.relu = torch.nn.ReLU(inplace=True)
            self.drop_layer = nn.Dropout(p=0.1)

        def make_layer(self, block, out_channels, blocks, stride=1):
            downsample = None
            kernel_size = (1,1,3)
            if (stride != 1) or (self.in_channels != out_channels):
                kernel_size=(1,1,13)
                downsample = torch.nn.Sequential(
                    conv3x3(self.in_channels, out_channels, stride=stride, kernel_size=kernel_size),
                    torch.nn.BatchNorm3d(out_channels))
            layers = []
            layers.append(block(self.in_channels, out_channels, kernel_size, stride, downsample))
            self.in_channels = out_channels
            for i in range(1, blocks):
                layers.append(block(out_channels, out_channels, kernel_size))
            return nn.Sequential(*layers)

        def forward(self, x):
            
            out = self.conv1(x)
            out = self.bn(out)
            out = self.relu(out)
            out = self.layer1(out) #residual layer 1
            out = self.relu(out)
            out = self.drop_layer(out)
            out = self.layer2(out) #residual layer 2
            out = self.relu(out)
            out = self.conv2(out)
            out = self.relu(out)
            out = self.conv3(out)

            return out.squeeze(1)
        
    return ResNet(ResidualBlock,layers=[2,1])



# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


'''
# only for classification, not working here.
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
'''