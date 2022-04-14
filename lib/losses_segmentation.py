import torch as torch
import torch.nn as nn
import torchvision.models
import numpy as np


def uniform_filter(kernel_size,n_channel,padding = None):
    if padding is None:
        uf = nn.Conv2d(n_channel,n_channel, kernel_size= kernel_size,  padding=int(np.round((kernel_size-1)/2)), bias=False)
    else:
        uf = nn.Conv2d(n_channel,n_channel, kernel_size= kernel_size,  padding=padding, bias=False)
    uf.weight.data = nn.Parameter(torch.ones_like(uf.weight,dtype=torch.float32),requires_grad = False)
    uf.weight.data = n_channel*uf.weight.data/torch.sum(uf.weight.data)
    return uf


class BCEBlurLoss(nn.Module):
    def __init__(self,kernel_size,channels):
        super(BCEBlurLoss, self).__init__()
        self.channels = channels
        self.blurrer = uniform_filter(kernel_size,len(channels))
        for param in self.blurrer.parameters():
            param.requires_grad = False
    def forward(self,target,prediction,code):
        target = target[:,self.channels,:,:]
        prediction = prediction[:,self.channels,:,:]
        #print(target.dtype)

        prediction = torch.sigmoid(prediction)
        return F.binary_cross_entropy(code*self.blurrer(prediction),target)


class JSBlurLoss(nn.Module): #keep in mind target = target / torch.sum(target,(2,3))
    def __init__(self,kernel_size,channels):
        super(JSBlurLoss, self).__init__()
        self.channels = channels
        self.blurrer = uniform_filter(kernel_size,len(channels),padding=kernel_size)

        for param in self.blurrer.parameters():
            param.requires_grad = False
        self.KLdiv = lambda p,q : torch.sum(torch.mean(p*torch.log(p/q),(0,1)))

    def forward(self,target,prediction,code):
        target = target[:,self.channels,:,:]
        prediction = prediction[:,self.channels,:,:]

        prediction = prediction.flatten(start_dim = 2)
        prediction =torch.softmax(prediction,2)
        prediction = prediction.view((-1,5,256,256))
        #print(prediction.shape)
        ##print(code.shape)
        #print(target.dtype)

        prediction = code*self.blurrer(prediction) + (1-code)*1/(prediction.shape[2]*prediction.shape[3])
        
        target = target + (1-code)

        target = code*self.blurrer(target/(torch.sum(target,(2,3)).unsqueeze(2).unsqueeze(3)))+(1-code)*1/(target.shape[2]*target.shape[3])
        prediction = prediction + 1e-10
        target = target + 1e-10
        M = .5*prediction + .5*target

        return .5*self.KLdiv(prediction,M) + .5*self.KLdiv(target,M)


class TotalLoss(nn.Module):
    def __init__(self,losses, #list of losses
                 weights):
        super(TotalLoss,self).__init__()

        self.losses = losses
        for i,L in enumerate(self.losses):
            self.add_module(str(i),L) #register loss functions

        self.weights = weights
    def forward(self,target,prediction,code):
        x=0
        for i, L in enumerate(self.losses):
            x = x + self.weights[i]*L(target,prediction,code)
        return x
# Use softmax across each layer instead of torch.sigmoid torch.softmax(predict, (2,3))
# have to take torch. mean (kl,dim = (0,1))
# Sinkhorn add this https://www.kernel-operations.io/geomloss/api/pytorch-api.html