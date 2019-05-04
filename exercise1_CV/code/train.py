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

epoch_shift = 0
parser = argparse.ArgumentParser()
parser.add_argument("--cont", action = "store_true", default = False)
parser.add_argument('-bsize', action="store_const", dest="batch_size", type=int, default=5)
parser.add_argument('-fbatch', action="store_const", dest="figure_batch", type=int, default=5)
parser.add_argument('-fepoch', action="store_const", dest="figure_epoch", type=int, default=10)
parser.add_argument('-epochs', action="store_const", dest="num_epochs", type=int, default=2000)
args = parser.parse_args()

# cuda & model init
print("initializing model, cuda ...")
cuda = torch.device('cuda')
model = ResNetModel(pretrained=False)
model.to(cuda)
# training loop init
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
loss_fn = torch.nn.MSELoss(reduction='mean')

train_loss = val_loss = 0
training_errors = []
validation_errors = []

# if flag --c is set, continute training from a previous snapshot
if(args.c):
    print("--c flag set")
    model.load_state_dict(torch.load(PATH))
    optimizer.load_state_dict(torch.load(OPATH))
    with open('results/training.errors', 'rb') as filehandle:
        training_errors = pickle.load(filehandle)
    with open('results/validation.errors', 'rb') as filehandle:
        validation_errors = pickle.load(filehandle)
    epoch_shift = len(training_errors) + 1
    print("resuming training from epoch {}".format(epoch_shift))



# get data loaders
train_loader = get_data_loader(args.batch_size, is_train=True)
val_loader = get_data_loader(args.batch_size, is_train=False)


for epoch in range(1,args.num_epochs):
    # if resuming training, update epoch #
    if(epoch<epoch_shift and args.c) :
        continue
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
    
    training_errors.append(train_loss/len(train_loader))
    print("epoch {}/{} : avg. training loss = {}   array size {}".format(epoch,args.num_epochs,training_errors[-1],len(training_errors)))
    
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
                 val_loss += torch.mean(loss).item()
                 # saving predictions from batch 5 every 10 epochs
                 if(idx==args.figure_batch and epoch%figure_epoch==0) :
                     # normalize keypoints to [0, 1] range
                    keypoints = normalize_keypoints(keypoints, img.shape)

                    # apply model
                    pred = model(img, '')

                    # show results
                    img_np = np.transpose(img.cpu().detach().numpy(), [0, 2, 3, 1])
                    img_np = np.round((img_np + 1.0) * 127.5).astype(np.uint8)
                    kp_pred = pred.cpu().detach().numpy().reshape([-1, 17, 2])
                    kp_gt = keypoints.cpu().detach().numpy().reshape([-1, 17, 2])
                    vis = weights.cpu().detach().numpy().reshape([-1, 17])

                    for bid in range(img_np.shape[0]):                
                        fig = plt.figure()
                        ax1 = fig.add_subplot(121)
                        ax2 = fig.add_subplot(122)
                        ax1.imshow(img_np[bid]), ax1.axis('off'), ax1.set_title('input + gt')
                        plot_keypoints(ax1, kp_gt[bid], vis[bid], img_size=img_np[bid].shape[:2], draw_limbs=True, draw_kp=True)
                        ax2.imshow(img_np[bid]), ax2.axis('off'), ax2.set_title('input + pred')
                        plot_keypoints(ax2, kp_pred[bid], vis[bid], img_size=img_np[bid].shape[:2], draw_limbs=True, draw_kp=True)
                        plt.savefig("fig_id{}_epoch{}.png".format(bid,epoch))
            print("validation loss : {}, MPJPE : {} pixels".format(val_loss,mpjpe/len(val_loader)))
            validation_errors.append(val_loss)

        print("saving snapshot @ epoch {}".format(epoch))
        print("training error # {}    validation error # {}".format(len(training_errors),len(validation_errors)))
        # save model state
        torch.save(model.state_dict(), PATH)
        torch.save(optimizer.state_dict(), OPATH)
        # store training/validation loss as binary data stream
        with open('results/training.errors', 'wb') as filehandle:  
            pickle.dump(training_errors, filehandle)
        with open('results/validation.errors', 'wb') as filehandle:  
            pickle.dump(validation_errors, filehandle)
