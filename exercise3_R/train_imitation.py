import sys
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import pickle
import torch

from imitation_learning.agent.bc_agent import BCAgent
from imitation_learning.dataloader import get_data_loader
from tensorboardX import SummaryWriter

datasets_dir = "./imitation_learning/data"
snapshot_dir = "./imitation_learning/snaps/snap"
tensorboard_dir="./imitation_learning/tensorboard"

# arg pars
parser = argparse.ArgumentParser()
parser.add_argument("--cont", action = "store_true", dest="continute_training", default = False, help ='continue training')
parser.add_argument("--snap", action = "store_true", dest="save_snaps", default = False, help='save snapshots every 5 epochs')
parser.add_argument('-name', type=str, dest="name", default="", help='name of the run')
parser.add_argument('-lr', type=float, dest="learning_rate", default=1e-3, help='learning rate')
parser.add_argument('-bsize', type=int, dest="batch_size", default=5, help='batch size')
parser.add_argument('-epochs', type=int, dest="num_epochs", default=2000, help='number of epochs')
# specific args for this training
parser.add_argument("--weighted", action = "store_true", dest="weighted", default = False, help ='apply weights to loss function')
parser.add_argument('-history', type=int, dest="history", default=1, help='number of previous frames to stack')

args = parser.parse_args()
print("settings :\ncontinute flag: {}\t batch size: {}".format(args.continute_training,args.batch_size))
print("number of epochs: {}\t name {}".format(args.num_epochs,args.name))
print("learning rate: {}\t save snapshots:{}".format(args.learning_rate,args.save_snaps))
print("weighted: {}\t\t history :{}".format(args.weighted,args.history))
snapshot_dir += args.name
# loaders
train_loader = get_data_loader(datasets_dir, frac=0.1, batch_size=args.batch_size, is_train=True , single_sample=False, history=args.history)
val_loader   = get_data_loader(datasets_dir, frac=0.1, batch_size=args.batch_size, is_train=False, single_sample=False, history=args.history)

# setting up cuda, agent
print("initializing agent, cuda ...")
cuda = torch.device('cuda')
agent = BCAgent(learning_rate=args.learning_rate, cuda=cuda, weighted=args.weighted, history=args.history)

# if flag --c is set, continute training from a previous snapshot
if(args.continute_training):
    print("--c flag set")
    try:
        agent.load(snapshot_dir)
    except FileNotFoundError:
        print("snapshot file(s) not found")

#tensorboard --logdir=./tensorboard --port=6006
print("starting tensorboard")
writer = SummaryWriter(os.path.join(tensorboard_dir,args.name))
# losses
loss_t = loss_v = 0
loss_e_t = loss_e_v = 0

print("training ...")
for epoch in range(1,args.num_epochs):    
    print("epoch {}/{}".format(epoch,args.num_epochs))
    val_iterator = iter(val_loader)
    
    for idx, (y_batch, X_batch) in enumerate(train_loader) :
        y_batch = y_batch.to(cuda)
        X_batch = X_batch.to(cuda)
        # one fwd/bwd pass
        temp = agent.update(X_batch,y_batch)
        loss_t += temp
        loss_e_t += temp

        if (idx+1)%10 ==0 :
            y_batch_val, X_batch_val = next(val_iterator)
            y_batch_val = y_batch_val.to(cuda)
            X_batch_val = X_batch_val.to(cuda)

            loss_v = agent.validate(X_batch_val,y_batch_val)
            loss_e_v += loss_v
            writer.add_scalar("training_loss", loss_t/10, epoch*len(train_loader)+idx)
            writer.add_scalar("validation_loss", loss_v, epoch*len(train_loader)+idx)
            loss_t = loss_v = 0

    writer.add_scalar("epoch_training_loss", loss_e_t/len(train_loader), epoch*(len(train_loader)+1))
    writer.add_scalar("epoch_validation_loss", loss_e_v/len(val_loader), epoch*(len(train_loader)+1))
    
    loss_e_t = loss_e_v = 0
    if args.save_snaps :
        agent.save(snapshot_dir)
    writer.close()