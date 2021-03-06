import torch as torch
import torch.nn as nn
import torchvision.models
import numpy as np
# Will make encoder, decoder nets.
# Network containing both encoder decoder

import torch.nn as nn
import torchvision.models

def convrelu(in_channels, out_channels, kernel, padding):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
    nn.ReLU(inplace=True),
  )


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_original_size0 = convrelu(3, 64, 3, 1) #convrelu(Num_channels, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)

        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())
        for l in self.base_layers:
            for param in l.parameters():
                param.requires_grad = False
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32) 

     

    def forward(self, input):
        input=input.float()
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        features0 = self.layer0(input)
        features1 = self.layer1(features0)
        features2 = self.layer2(features1)
        features3 = self.layer3(features2)
        features4 = self.layer4(features3)
        

        return x_original,features0,features1,features2,features3,features4

class Decoder(nn.Module):
    def __init__(self, n_class=10):
        super().__init__()

        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.conv_final = convrelu(64 + 128, 64, 3, 1) #        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        #now Num_channels=1 not Num_channels=3
        # labels are 256 x 256 OR you could resize the mask 
        

        self.conv_last = nn.Conv2d(64,n_class, 1)
    
    def forward(self, x_original,layer0,layer1,layer2,layer3,layer4):
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_final(x) # x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

class EncDecResNetUNet(nn.Module):
    def __init__(self,n_class=10):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(n_class = n_class)
    
    def forward(self, input):
        x_original,features0,features1,features2,features3,features4 = self.encoder(input)
        out = self.decoder(x_original,features0,features1,features2,features3,features4)
        
        return out