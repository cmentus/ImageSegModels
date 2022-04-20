import torch as torch
import torch.nn as nn
import torchvision.models
import numpy as np
import time

def train_model(dataloaders, 
                model, 
                num_epochs,
                lossfn, # function form loss(target,prediction,code)
                metrics, # dictionary string : f(target,prediction,code) #print metrics, record in dict -> list, plot.
                optimizer, 
                scheduler, 
                visualize, # plot the mask, ground truth etc. visualize(input,prediction,target,epoch,dir) (done)
                plot_epoch = 20,
                savedir=''):
    
    best_loss = 1e10

    train_metrics = {k:[] for k in metrics.keys()} # defines dict metric_key -> []
    val_metrics = {k:[] for k in metrics.keys()} # stores the meman metric over each epoch


    for epoch in range(num_epochs):
        since = time.time()

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()  
            else:
                model.eval()
            epoch_metrics = {k: 0. for k in metrics.keys()}
            for img, code, mask, contour in dataloaders[phase]: # target 10 dim tensor with mask and contour
                img = img.to(device) 
                code = code.to(device)
                mask = mask.to(device)
                contour = contour.to(device)
                

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    prediction = model(img) # forward pass. model(inputs) calls model.forward(inputs)
                    #print(type(prediction))
                    target = torch.concat([mask,contour],1) #concat along channel dim
                    loss = lossfn(target,prediction,code)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward() # back prop. to calc gradient.
                        optimizer.step()

                # update metrics:
                with torch.no_grad():
                    for k, metric_loss in metrics.items():
                        epoch_metrics[k] += metric_loss(target,prediction,code)/len(dataloaders[phase]) #divide by number batches
                
            if phase == 'train':
                scheduler.step()

            model.eval() # Turn on eval for calculating metrics, visualizing, etc
            outputs = []

            for k in epoch_metrics.keys():
                outputs.append("{}: {:4f}".format(k, epoch_metrics[k]))
            print("{}: {}".format(phase, ", ".join(outputs)))


            if phase == 'train':
                for k in train_metrics.keys():
                    train_metrics[k] = train_metrics[k] + [epoch_metrics[k]] # adding list is concatenating to the end.
            
            if phase == 'val':
                for k in val_metrics.keys():
                    val_metrics[k] = val_metrics[k] + [epoch_metrics[k]] # adding list is concatenating to the end.
            
            
            # visualization:
            if phase == 'val':
                if epoch % plot_epoch == 0:
                    #vis_img = img.cpu()[0]
                    #vis_prediction = torch.sigmoid(prediction.cpu()[0])
                    #vis_mask = mask.cpu()[0]
                    #vis_contour = contour.cpu()[0]
                    #visualize(vis_img,vis_prediction,vis_mask,
                    #          vis_contour,epoch=epoch,dir=savedir)
                    #visualize should be a function of input-img, model, ground truth.
                    #this way we have flexibility to visualize attention fields coming 
                    #from the model etc.
                    #Within the visualization function predictions are made instead of before.

                    for k,v in train_metrics.items():
                        f, ax = plt.subplots(figsize = (5,5))
                        ax.plot(v,label = 'train')
                        ax.set_xlabel('epochs',fontsize = 10)
                        ax.set_ylabel(k,fontsize = 10)
                        ax.set_title(k)
                        ax.plot(val_metrics[k],label ='val')
                        ax.set_title(k)
                        f.legend()
                        #f.savefig(savedrive +k+ hyperstring+str(epoch)+'.png') #if overriding then remove epoch
                        f.savefig(savedir+k+'.png') #if overriding then remove epoch
                

    return model,train_metrics,val_metrics