import torch
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from torch import nn
from pdb import set_trace
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import seaborn as sns
import pandas as pd
    

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


def show_images_rgb(inputs, outputs, num_images= 5, grayscale=False):
    
    fig = plt.figure(figsize = (20, 14))
    rows = 4
    j=0
    if not grayscale:
        for i in range(0, rows*num_images):
            if i >=0 and i <num_images:
                fig.add_subplot(rows, num_images, i+1)
                plt.imshow(inputs[i%num_images].permute(1,2,0))
            
            if i >= num_images and i < 2*num_images:
                fig.add_subplot(rows, num_images, i+1)
                plt.imshow(outputs[i%num_images,:3].permute(1,2,0))
            
            if i >= 2*num_images and i < 3*num_images:
                fig.add_subplot(rows, num_images, i+1)
                plt.imshow(outputs[i%num_images,3:].permute(1,2,0))
    else:
        for i in range(0, rows*num_images):
            if i >=0 and i <num_images:
                fig.add_subplot(rows, num_images, i+1)
                plt.imshow(inputs[i%num_images,0], cmap='gray')
            
            if i >= num_images and i < 2*num_images:
                fig.add_subplot(rows, num_images, i+1)
                plt.imshow(outputs[i%num_images,0], cmap='gray')
            
            if i >= 2*num_images and i < 3*num_images:
                fig.add_subplot(rows, num_images, i+1)
                plt.imshow(outputs[i%num_images,1], cmap='gray')


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

def show_outputs_rgb(TransformerOutputs, RNNOutputs, targets, inputs, num_images= 4, gray = False):
    
    fig = plt.figure(figsize = (40, 30))
    rows = 7
    criterion = torch.nn.MSELoss()
    ssim = SSIM()

    if not gray:
        for i in range(0, rows*num_images):
            if i >=0 and i < num_images:
                fig.add_subplot(rows, num_images, i+1).axis('off')
                plt.imshow(inputs[i%num_images].permute(1,2,0).detach().cpu())
            
            if i >= num_images and i < 2*num_images:
                fig.add_subplot(rows, num_images, i+1).axis('off')
                plt.imshow(TransformerOutputs[i%num_images,:3].permute(1,2,0).detach().cpu())
            
            if i >= 2*num_images and i < 3*num_images:
                fig.add_subplot(rows, num_images, i+1).axis('off')
                plt.imshow(TransformerOutputs[i%num_images,3:].permute(1,2,0).detach().cpu())
                        
            if i >= 3*num_images and i < 4*num_images:
                fig.add_subplot(rows, num_images, i+1).axis('off')
                plt.imshow(RNNOutputs[i%num_images,0].detach().cpu(), cmap='gray')
            
            if i >= 4*num_images and i < 5*num_images:
                fig.add_subplot(rows, num_images, i+1).axis('off')
                plt.imshow(RNNOutputs[i%num_images,1].detach().cpu(), cmap='gray')
                
            if i >= 5*num_images and i < 6*num_images:
                fig.add_subplot(rows, num_images, i+1).axis('off')
                plt.imshow(targets[i%num_images, :3].permute(1,2,0).detach().cpu())
            
            if i >= 6*num_images and i < 7*num_images:
                fig.add_subplot(rows, num_images, i+1).axis('off')
                plt.imshow(targets[i%num_images, 3:].permute(1,2,0).detach().cpu())
    else:
        for i in range(0, rows*num_images):
            if i >=0 and i < num_images:
                fig.add_subplot(rows, num_images, i+1)
                if i ==0:
                    fig.axes[i].set_ylabel('Inputs')
                plt.imshow(inputs[i%num_images,0].detach().cpu())
            
            if i >= num_images and i < 2*num_images:
                fig.add_subplot(rows, num_images, i+1)
                if i == num_images:
                    fig.axes[i].set_ylabel('Transformer o/p1')
                plt.imshow(TransformerOutputs[i%num_images,0].detach().cpu())
                loss1 = criterion(TransformerOutputs[i%num_images,0].detach().cpu(), targets[i%num_images, 0].detach().cpu())
                ssim1 = ssim(TransformerOutputs[i%num_images,0].reshape(1,1,28,28).detach().cpu(), targets[i%num_images, 0].reshape(1,1,28,28).detach().cpu())
                ssim2 = ssim(TransformerOutputs[i%num_images,0].reshape(1,1,28,28).detach().cpu(), targets[i%num_images, 1].reshape(1,1,28,28).detach().cpu())
                loss2 = criterion(TransformerOutputs[i%num_images,0].detach().cpu(), targets[i%num_images, 1].detach().cpu())
                if loss1< loss2:
                    fig.axes[i].set_xlabel('MSE: '+str(loss1.item())+'\nSSIM: '+str(ssim1.item()))
                else:
                    fig.axes[i].set_xlabel('MSE: '+str(loss2.item())+'\nSSIM: '+str(ssim2.item()))
                
            
            if i >= 2*num_images and i < 3*num_images:
                fig.add_subplot(rows, num_images, i+1)
                if i == 2*num_images:
                    fig.axes[i].set_ylabel('Transformer o/p2')
                plt.imshow(TransformerOutputs[i%num_images,1].detach().cpu())
                loss1 = criterion(TransformerOutputs[i%num_images,1].detach().cpu(), targets[i%num_images, 0].detach().cpu())
                ssim1 = ssim(TransformerOutputs[i%num_images,1].reshape(1,1,28,28).detach().cpu(), targets[i%num_images, 0].reshape(1,1,28,28).detach().cpu())
                ssim2 = ssim(TransformerOutputs[i%num_images,1].reshape(1,1,28,28).detach().cpu(), targets[i%num_images, 1].reshape(1,1,28,28).detach().cpu())
                loss2 = criterion(TransformerOutputs[i%num_images,1].detach().cpu(), targets[i%num_images, 1].detach().cpu())
                if loss1< loss2:
                    fig.axes[i].set_xlabel('MSE: '+str(loss1.item())+'\nSSIM: '+str(ssim1.item()))
                else:
                    fig.axes[i].set_xlabel('MSE: '+str(loss2.item())+'\nSSIM: '+str(ssim2.item()))
            
            if i >= 3*num_images and i < 4*num_images:
                fig.add_subplot(rows, num_images, i+1)
                if i == 3*num_images:
                    fig.axes[i].set_ylabel('RNN o/p1')
                plt.imshow(RNNOutputs[i%num_images,0].detach().cpu())
                loss1 = criterion(RNNOutputs[i%num_images,0].detach().cpu(), targets[i%num_images, 0].detach().cpu())
                ssim1 = ssim(RNNOutputs[i%num_images,0].reshape(1,1,28,28).detach().cpu(), targets[i%num_images, 0].reshape(1,1,28,28).detach().cpu())
                ssim2 = ssim(RNNOutputs[i%num_images,0].reshape(1,1,28,28).detach().cpu(), targets[i%num_images, 1].reshape(1,1,28,28).detach().cpu())
                loss2 = criterion(RNNOutputs[i%num_images,0].detach().cpu(), targets[i%num_images, 1].detach().cpu())
                if loss1< loss2:
                    fig.axes[i].set_xlabel('MSE: '+str(loss1.item())+'\nSSIM: '+str(ssim1.item()))
                else:
                    fig.axes[i].set_xlabel('MSE: '+str(loss2.item())+'\nSSIM: '+str(ssim2.item()))
            
            if i >= 4*num_images and i < 5*num_images:
                fig.add_subplot(rows, num_images, i+1)
                if i == 4*num_images:
                    fig.axes[i].set_ylabel('RNN o/p2')
                plt.imshow(RNNOutputs[i%num_images,1].detach().cpu())
                loss1 = criterion(RNNOutputs[i%num_images,1].detach().cpu(), targets[i%num_images, 0].detach().cpu())
                ssim1 = ssim(RNNOutputs[i%num_images,1].reshape(1,1,28,28).detach().cpu(), targets[i%num_images, 0].reshape(1,1,28,28).detach().cpu())
                ssim2 = ssim(RNNOutputs[i%num_images,1].reshape(1,1,28,28).detach().cpu(), targets[i%num_images, 1].reshape(1,1,28,28).detach().cpu())
                loss2 = criterion(RNNOutputs[i%num_images,1].detach().cpu(), targets[i%num_images, 1].detach().cpu())
                if loss1< loss2:
                    fig.axes[i].set_xlabel('MSE: '+str(loss1.item())+'\nSSIM: '+str(ssim1.item()))
                else:
                    fig.axes[i].set_xlabel('MSE: '+str(loss2.item())+'\nSSIM: '+str(ssim2.item()))
            
            if i >= 5*num_images and i < 6*num_images:
                fig.add_subplot(rows, num_images, i+1)
                if i == 5*num_images:
                    fig.axes[i].set_ylabel('Target 1')
                plt.imshow(targets[i%num_images, 0].detach().cpu())
                
                
            if i >= 6*num_images and i < 7*num_images:
                fig.add_subplot(rows, num_images, i+1)
                if i == 6*num_images:
                    fig.axes[i].set_ylabel('Target 2')
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


def fetch_mse_ssim(dataset, model, transformer, sparse = True):
    
    mse = nn.MSELoss()
    ssim = SSIM()
    # If you want to only choose a few samples
    # Make sure dataset has a batchsize greater than 1
    if sparse:

        # Check if dataset has greater than batch size one
        mse_list, ssim_list = [], []
        for i, (inputs, targets) in enumerate(dataset):
            if transformer:
                outputs = model(inputs)
            else:
                outputs = model(inputs, flag = False)
                outputs = torch.cat([outputs[1].reshape(-1,1,28,28), outputs[2].reshape(-1,1,28,28)], axis = 1)
            
            loss1 = mse(outputs[0,1].detach().cpu(), targets[0, 0].detach().cpu())
            ssim1 = ssim(outputs[0,1].reshape(1,1,28,28).detach().cpu(), targets[0, 0].reshape(1,1,28,28).detach().cpu())
            ssim2 = ssim(outputs[0,1].reshape(1,1,28,28).detach().cpu(), targets[0, 1].reshape(1,1,28,28).detach().cpu())
            loss2 = mse(outputs[0,1].detach().cpu(), targets[0, 1].detach().cpu())
            if ssim1>ssim2:
                mse_list.append(loss1.item())
                ssim_list.append(ssim1.item())
            else:
                mse_list.append(loss2.item())
                ssim_list.append(ssim2.item())
    
    return mse_list, ssim_list



def sns_plotter(mseDict, ssimDict, savefig=False):


    # Convert the dictionary into a DataFrame
    df_mse  = pd.DataFrame(mseDict)
    df_ssim = pd.DataFrame(ssimDict)

    # Set up the plot
    # Change plotting parameters
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times New Roman"
    })
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")  # You can choose different styles

    # Create the boxplot
    ax = sns.boxplot(data=df_mse, palette="Set3")

    # Add title and labels
    plt.title("MSE", fontsize=25)


    # Customize further if needed (e.g., add grid, adjust font sizes, etc.)
    plt.xticks(rotation=0, fontsize=20) 
    plt.yticks(fontsize=14)

    # Save the plot (optional)
    if savefig:
        if not os.path.isdir('./figures'):
            print('figures path does not exist. Making it..')
            os.mkdir('figures')
        
        plt.savefig("./figures/boxplot_mse.png", dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()