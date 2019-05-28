import torch
import pickle
import numpy as np
import os
import gzip
from PIL import Image
from imitation_learning.utils import *
import torch.utils.data

PKG_NAME = "data.pkl.gzip"

class PickleReader(torch.utils.data.Dataset):
    def __init__(self, datasets_dir, frac=0.1, single_sample=False, is_train=True):
        
        print("decompressing data...")

        data_file = os.path.join(datasets_dir, PKG_NAME)
  
        f = gzip.open(data_file,'rb')
        data = pickle.load(f)

        self.is_train = is_train

        # get images as features and actions as targets
        X = np.array(data["state"]).astype('uint8')
        y = np.array(data["action"]).astype('float32')

        # get training or validation set depending on is_train
        self.n_samples = len(data["state"])

        if self.is_train:
            self.X, self.y = X[:int((1-frac) * self.n_samples)], y[:int((1-frac) * self.n_samples)]
        else :
            self.X, self.y = X[int((1-frac) * self.n_samples):], y[int((1-frac) * self.n_samples):]

        del X,y,data
        print("X shape {}".format(self.X.shape))
        print("y shape {}".format(self.y.shape))
        print("y len {}".format(len(self.y)))
        self.single_sample = single_sample

    def __getitem__(self, idx):
        if self.single_sample:
            idx = 666
        
        label = self.y[idx,:]
        sample = self.X[idx,:]

        # preprocess data
        label = action_to_id(label)
        label = torch.Tensor(label)
        sample = rgb2gray(sample)
        sample = sample.transpose(2,0,1)
        sample = torch.from_numpy(sample)
        sample = torch.transpose(sample)
        print("l shape : {}".format(label.shape))
        print("s shape : {}".format(sample.shape))
        return label, sample
        

    def __len__(self):
        return len(self.y)


def get_data_loader(datasets_dir, frac=0.1, batch_size=1, is_train=False, single_sample=False):

    reader = PickleReader(datasets_dir, frac, single_sample=single_sample, is_train=is_train)
    
    data_loader = torch.utils.data.DataLoader(reader,
                                              batch_size=batch_size,
                                              shuffle=is_train,
                                              num_workers=4 if is_train else 1)
    return data_loader