import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

from model.model import ResNetModel
from model.data import get_data_loader
from utils.plot_util import plot_keypoints
import torch
import argparse

PATH = "model/fuckme.pth"
OPATH = "model/fuckyou.pth"
batch_size = 10
num_epochs = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--c", action = "store_true")
args = parser.parse_args()

# cuda
cuda = torch.device('cuda')
model = ResNetModel(pretrained=False)
#model.load_state_dict(torch.load(PATH_TO_CKPT))
model.to(cuda)
# training loop init
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
loss_fn = torch.nn.MSELoss(reduction='elementwise_mean')

if(args.c):
    model.load_state_dict(torch.load(PATH))
    optimizer.load_state_dict(torch.load(OPATH))


train_loader = get_data_loader(batch_size, is_train=True)
val_loader = get_data_loader(batch_size, is_train=False)
train_loss = eval_loss = 0

for epoch in range(num_epochs):
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
    print("epoch {0}/{1}".format(epoch,num_epochs))
    print("avg. training loss : {}".format(train_loss/len(train_loader)))

    if epoch % 5 == 0: 
        with torch.no_grad():
            model.eval()
            eval_loss = 0
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
            print("evaluation loss : {}".format(eval_loss))
            print("MPJPE : {}".format(mpjpe/len(val_loader)))

        torch.save(model.state_dict(), PATH)
        torch.save(optimizer.state_dict(), OPATH)
