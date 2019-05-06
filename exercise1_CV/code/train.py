import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch

from model.model import ResNetModel
from model.data import get_data_loader
from utils.plot_util import plot_keypoints
from run_forward import normalize_keypoints

import argparse
import pickle

PATH = "model/fuckme.pth"
OPATH = "model/fuckyou.pth"

epoch_shift = 0
parser = argparse.ArgumentParser()
parser.add_argument("--cont", action = "store_true", dest="continute_training", default = False)
parser.add_argument("--snap", action = "store_true", dest="save_snaps", default = False)
parser.add_argument('-lr', type=float, dest="learning_rate", default=1e-4)
parser.add_argument('-bsize', type=int, dest="batch_size", default=5)
parser.add_argument('-fbatch', type=int, dest="figure_batch", default=5)
parser.add_argument('-fepoch', type=int, dest="figure_epoch", default=10)
parser.add_argument('-epochs', type=int, dest="num_epochs", default=2000)
args = parser.parse_args()
print("settings :\ncontinute flag: {}\t batch size: {}\t plot batch #: {}".format(args.continute_training,args.batch_size,args.figure_batch))
print("plot every {} epochs\t number of epochs: {}".format(args.figure_epoch,args.num_epochs))
print("learning rate: {}\t save snapshots:{}".format(args.learning_rate,args.save_snaps))
# cuda & model init
print("initializing model, cuda ...")
cuda = torch.device('cuda')
model = ResNetModel(pretrained=False)
model.to(cuda)
# training loop init
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
loss_fn = torch.nn.MSELoss(reduction='none')

train_loss = val_loss = 0
training_errors = []
validation_errors = []
mean_pixel_errors = [] #MPJPE over epochs
# if flag --c is set, continute training from a previous snapshot
if(args.continute_training):
    print("--c flag set")
    try:
        model.load_state_dict(torch.load(PATH))
        optimizer.load_state_dict(torch.load(OPATH))
        with open('results/training.errors', 'rb') as filehandle:
            training_errors = pickle.load(filehandle)
        with open('results/validation.errors', 'rb') as filehandle:
            validation_errors = pickle.load(filehandle)
        with open('results/pixel.errors', 'rb') as filehandle:
            mean_pixel_errors = pickle.load(filehandle)
    except FileNotFoundError:
        print("snapshot file(s) not found")

    epoch_shift = len(training_errors) + 1
    print("resuming training from epoch {}".format(epoch_shift))



# get data loaders
train_loader = get_data_loader(args.batch_size, is_train=True)
val_loader = get_data_loader(args.batch_size, is_train=False)

print("training ...")
for epoch in range(1,args.num_epochs):
    # if resuming training, update epoch #
    if(epoch<epoch_shift and args.continute_training) :
        continue
    model.train()
    train_loss=0
    for idx, (img, keypoints, weights) in enumerate(train_loader):
        optimizer.zero_grad()
        img = img.to(cuda)
        keypoints = keypoints.to(cuda)
        weights = weights.to(cuda)
        output = model(img,'')
        print((weights.repeat_interleave(2).float()).shape)
        loss = loss_fn(output, keypoints)*(weights.repeat_interleave(2).float())
        loss = torch.sum(loss)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    training_errors.append(train_loss/len(train_loader))
    print("epoch {}/{} : avg. training loss = {}".format(epoch,args.num_epochs,training_errors[-1]))
    
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
                loss = loss_fn(output, keypoints)*(weights.repeat_interleave(2).float())
                visible = torch.sum(weights>0.5).item()
                mpjpe += (torch.sum(torch.sqrt(loss))/visible).item()
                val_loss += torch.sum(loss).item()
                # saving predictions from batch 5 every 10 epochs
                if(idx==args.figure_batch and epoch%args.figure_epoch==0) :
                    # normalize keypoints to [0, 1] range
                    keypoints = normalize_keypoints(keypoints, img.shape)
                    
                    # apply model
                    pred = model(img, '')
                    pred = normalize_keypoints(pred, img.shape)
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
                        plt.savefig("results/fig_id{}_epoch{}.png".format(bid,epoch))
                        plt.clf()
                        plt.close()
                    plt.close('all')

            validation_errors.append(val_loss/len(val_loader))
            mean_pixel_errors.append(mpjpe/len(val_loader))
            # add later : val_loss/len(val_loader)
            print("validation loss : {}, MPJPE : {} pixels".format(validation_errors[-1] ,mean_pixel_errors[-1]))

        if args.save_snaps:
            print("saving snapshot @ epoch {}".format(epoch))
            # save model state
            torch.save(model.state_dict(), PATH)
            torch.save(optimizer.state_dict(), OPATH)
            # store training/validation loss as binary data stream
            with open('results/training.errors', 'wb') as filehandle:  
                pickle.dump(training_errors, filehandle)
            with open('results/validation.errors', 'wb') as filehandle:  
                pickle.dump(validation_errors, filehandle)
            with open('results/pixel.errors', 'wb') as filehandle:  
                pickle.dump(mean_pixel_errors, filehandle)
