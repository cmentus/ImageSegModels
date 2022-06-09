from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms, datasets, models
import random
import numpy as np

def grad_contour(mask):
    #Takes a torch array of size N_data x Channels x Height x Width
    #Return the contour using square of gradient.
    return(torch.square((mask[:,:,:-1,:-1]-mask[:,:,1:,:-1]))+torch.square((mask[:,:,:-1,:-1]-mask[:,:,:-1,1:])))

class ImageLabelDataset(Dataset):
    def __init__(self, images, codes, labels, transform=None,rand_rot=False):
    #self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)

        self.permutation = np.random.permutation(images.shape[0])
        self.input_images = images[self.permutation,:,:]
        self.codes = codes[self.permutation,:]
        self.target_masks = labels[self.permutation,:,:,:]
        self.transform = transform
        self.rand_rot = rand_rot
        self.rotate = transforms.RandomRotation(180)
        self.upsample = transforms.Resize((256,256))     #nn.Upsample(size=(256,256), mode='bilinear', align_corners=True)


    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx,expand=True,):
        image = self.input_images[idx,:,:]
        code = self.codes[idx,:,:,:]
        mask = self.target_masks[idx,:,:,:]
        if self.rand_rot:
            img_mask = self.rotate(torch.concat([image.unsqueeze(0),mask],dim=0))
            image = img_mask[0,:,:]
            mask = img_mask[-5:,:,:]

        if self.transform is not None:
            image = self.transform(image)
        if expand:
            image = image.unsqueeze(0).expand(3,-1,-1) #image was already resized
            #this makes it 3x64x64
        else:
            image = image

        contour = torch.square((mask[:,:-1,:-1]-mask[:,1:,:-1]))+torch.square((mask[:,:-1,:-1]-mask[:,:-1,1:]))
        contour = (contour>0).type(torch.float32)
        contour = self.upsample(contour.unsqueeze(0))[0]
        
        return image, code, mask, contour
    def enable_flipping(self):
        self.rand_rot = True
    def disable_flipping(self):
        self.rand_rot = False