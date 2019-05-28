from __future__ import print_function

import sys
sys.path.append("../") 

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import pickle
import torch._C as torch

from exercise3_R.imitation_learning.agent.bc_agent import BCAgent
from exercise3_R.tensorboard_evaluation import Evaluation
from exercise3_R.imitation_learning.dataloader import get_data_loader

'''
mini batch shit :
def minibatched(data: np.ndarray, batch_size: int) -> List[np.ndarray]:
    assert len(data) % batch_size == 0, ("Data length {} is not multiple of batch size {}"
                                         .format(len(data), batch_size))
    return data.reshape(-1, batch_size, *data.shape[1:])

training loop :
    ix = np.arange(len(x_train))
    
    for epoch in range(num_epochs):
        print("Epoch {} / {}:".format(epoch + 1, num_epochs))
        training_predictions = []
        
        np.random.shuffle(ix)
        x_train_batched = minibatched(x_train[ix], batch_size)
        y_train_batched = minibatched(y_train[ix], batch_size)
'''


    # preprocess data
    

    # train model (you can change the parameters!)

datasets_dir = "./data"
snapshot_dir = "./snaps/snap"
tensorboard_dir="./tensorboard"

# arg pars
parser = argparse.ArgumentParser()
parser.add_argument("--cont", action = "store_true", dest="continute_training", default = False, help ='continue training')
parser.add_argument("--snap", action = "store_true", dest="save_snaps", default = False, help='save snapshots every 5 epochs')
parser.add_argument('-lr', type=float, dest="learning_rate", default=1e-3, help='learning rate')
parser.add_argument('-bsize', type=int, dest="batch_size", default=5, help='batch size')
parser.add_argument('-epeval', type=int, dest="epoch_eval", default=1, help='evaluate every x epochs') 
parser.add_argument('-epochs', type=int, dest="num_epochs", default=2000, help='number of epochs')
args = parser.parse_args()
print("settings :\ncontinute flag: {}\t batch size: {}".format(args.continute_training,args.batch_size))
print("number of epochs: {}\t evaluate every {} epochs".format(args.num_epochs,args.epoch_eval))
print("learning rate: {}\t save snapshots:{}".format(args.learning_rate,args.save_snaps))

# loaders
train_loader = get_data_loader(datasets_dir, frac=0.1, batch_size=args.batch_size, is_train=True, single_sample=False)
val_loader = get_data_loader(datasets_dir, frac=0.1, batch_size=args.batch_size, is_train=False, single_sample=False)

# losses
train_loss = val_loss = 0


# getting cuda, agent
print("initializing agent, cuda ...")
agent = BCAgent(learning_rate=args.learning_rate)
cuda = torch.device('cuda')
agent.net.to(cuda)

#tensorboard --logdir=path/to/log-directory --port=6006

# if flag --c is set, continute training from a previous snapshot
if(args.continute_training):
    print("--c flag set")
    try:
        agent.load(snapshot_dir)
    except FileNotFoundError:
        print("snapshot file(s) not found")

print("starting tensorboard")
tensorboard_eval = Evaluation(name="eval" ,store_dir=tensorboard_dir, stats= ['train_loss', 'val_loss'])


print("training ...")
for epoch in range(1,args.num_epochs):    
    print("epoch {}/{}".format(epoch,args.num_epochs))
    val_iterator = iter(val_loader)
    
    for idx, (y_batch, X_batch) in enumerate(train_loader) :
        y_batch = y_batch.to(cuda)
        X_batch = X_batch.to(cuda)
        # one f/b pass
        loss_t = + agent.update(X_batch,y_batch)

        if idx%10 ==0 :
            X_batch_val, y_batch_val = next(val_iterator)
            y_batch_val = y_batch_val.to(cuda)
            X_batch_val = X_batch_val.to(cuda)
            loss_v = agent.validate(X_batch_val,y_batch_val)
            eval_dict = dict()
            eval_dict['train_loss'] = loss_t/10
            eval_dict['val_loss'] = loss_v
            loss_t = loss_v =0
            tensorboard_eval.write_episode_data(idx, eval_dict)
            if args.save_snaps :
                agent.save(snapshot_dir)
