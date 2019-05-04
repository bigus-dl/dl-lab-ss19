import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch

from model.model import ResNetModel
from model.data import get_data_loader
from utils.plot_util import plot_keypoints

import argparse
import pickle

PATH = "model/fuckme.pth"
OPATH = "model/fuckyou.pth"
batch_size = 5
num_epochs = 2000

parser = argparse.ArgumentParser()
parser.add_argument("--c", action = "store_true")
args = parser.parse_args()

# cuda & model init
cuda = torch.device('cuda')
model = ResNetModel(pretrained=False)
model.to(cuda)
# training loop init
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
loss_fn = torch.nn.MSELoss(reduction='elementwise_mean')

train_loss = val_loss = 0
training_errors = validation_errors = []

# if flag -c is set, continute training from a previous snapshot
if(args.c):
    model.load_state_dict(torch.load(PATH))
    optimizer.load_state_dict(torch.load(OPATH))
    with open('training.errors', 'rb') as filehandle:
        training_errors = pickle.load(filehandle)
    with open('validation.errors', 'rb') as filehandle:
        training_errors = pickle.load(filehandle)
    print("resuming training from epoch {}".format(len(training_errors)))

# get data loaders
train_loader = get_data_loader(batch_size, is_train=True)
val_loader = get_data_loader(batch_size, is_train=False)


for epoch in range(num_epochs):
    # if resuming training, update epoch #
    if(epoch==0 & args.c) :
        epoch = len(training_errors) + 1
    model.train()
    train_loss=0
    for idx, (img, keypoints, weights) in enumerate(train_loader):
        optimizer.zero_grad()
        img = img.to(cuda)
        keypoints = keypoints.to(cuda)
        weights = weights.to(cuda)
        output = model(img,'')
        loss = loss_fn(keypoints, output)*(weights.repeat_interleave(2).float())
        loss = torch.mean(loss)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (idx>=5) :
            break
    print("epoch {0}/{1}".format(epoch+1,num_epochs))
    training_errors.append(train_loss/len(train_loader))
    print("avg. training loss : {}".format(training_errors[-1]))
    
    if epoch % 5 == 0: 
        with torch.no_grad():
            model.eval()
            val_loss = 0
            mpjpe = 0
            for idx, (img, keypoints, weights) in enumerate(val_loader):
                 img = img.to(cuda)
                 keypoints = keypoints.to(cuda)
                 weights = weights.to(cuda)
                 output = model(img, '')
                 loss = loss_fn(keypoints, output)*(weights.repeat_interleave(2).float())
                 mpjpe += torch.mean(torch.sqrt(loss)).item()
                 eval_loss += torch.mean(loss).item()
                 if(idx>=5) :
                    break 
            print("validation loss : {}".format(val_loss))
            validation_errors.append(eval_loss)
            print("MPJPE : {} pixels".format(mpjpe/len(val_loader)))

        # save model state
        torch.save(model.state_dict(), PATH)
        torch.save(optimizer.state_dict(), OPATH)
        # store training/validation loss as binary data stream
        with open('results/training.errors', 'wb') as filehandle:  
            pickle.dump(training_errors, filehandle)
        with open('results/validation.errors', 'wb') as filehandle:  
            pickle.dump(validation_errors, filehandle)
