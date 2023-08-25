import torch
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from torch import nn
from pdb import set_trace
import matplotlib.pyplot as plt

    

def show_images(inputs, outputs, num_images= 5):
    
    fig = plt.figure(figsize = (20, 14))
    rows = 4
    j=0
    for i in range(0, rows*num_images):
        if i >=0 and i <num_images:
            fig.add_subplot(rows, num_images, i+1)
            plt.imshow(inputs[i%num_images].squeeze(0))
        
        if i >= num_images and i < 2*num_images:
            fig.add_subplot(rows, num_images, i+1)
            plt.imshow(outputs[i%num_images, 0])
        
        if i >= 2*num_images and i < 3*num_images:
            fig.add_subplot(rows, num_images, i+1)
            plt.imshow(outputs[i%num_images, 1])

#Print only the output images
def show_outputs(outputs, targets, inputs, num_images= 4):
    
    fig = plt.figure(figsize = (20, 14))
    rows = 5
    for i in range(0, rows*num_images):
        if i >=0 and i < num_images:
            fig.add_subplot(rows, num_images, i+1)
            plt.imshow(inputs[i%num_images,0].detach().cpu())
        
        if i >= num_images and i < 2*num_images:
            fig.add_subplot(rows, num_images, i+1)
            plt.imshow(outputs[i%num_images,0].detach().cpu())
        
        if i >= 2*num_images and i < 3*num_images:
            fig.add_subplot(rows, num_images, i+1)
            plt.imshow(outputs[i%num_images, 1].detach().cpu())
        
        if i >= 3*num_images and i < 4*num_images:
            fig.add_subplot(rows, num_images, i+1)
            plt.imshow(targets[i%num_images, 0].detach().cpu())
            
        if i >= 4*num_images and i < 5*num_images:
            fig.add_subplot(rows, num_images, i+1)
            plt.imshow(targets[i%num_images, 1].detach().cpu())



class Downsample(object):
    def __init__(self, size=[1,196]):
        self.size=size

    def __call__(self, tensor):
        img = np.squeeze(tensor)
        m = torch.nn.AvgPool2d(2, stride=2)
        return m(img.unsqueeze(0))

    def __repr__(self):
        return self.__class__.__name__+'({})'.format(self.size)

class Noise(object):
    def __init__(self, mean=0, dev=1):
        self.mean = mean
        self.dev = dev
    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size())*self.dev + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + 'mean = {0}, dev= {1}', format(self.mean, self.dev)
    



import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)