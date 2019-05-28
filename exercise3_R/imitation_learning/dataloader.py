import torch
from torchvision import transforms
import pickle
import numpy as np
import os
import gzip
from PIL import Image
from imitation_learning.utils import *

PKG_NAME = "data.pkl.gzip"

class PickleReader:
    def __init__(self, datasets_dir, frac=0.1, single_sample=False, is_train=True):
        
        print("decompressing data...")

        data_file = os.path.join(datasets_dir, PKG_NAME)
  
        f = gzip.open(data_file,'rb')
        data = pickle.load(f)

        self.is_train = is_train

        # get images as features and actions as targets
        X = np.array(data["state"]).astype('float32')
        y = np.array(data["action"]).astype('float32')

        # get training or validation set depending on is_train
        self.n_samples = len(data["state"])

        if self.is_train:
            self.X, self.y = X[:int((1-frac) * self.n_samples)], y[:int((1-frac) * self.n_samples)]
        else :
            self.X, self.y = X[int((1-frac) * self.n_samples):], y[int((1-frac) * self.n_samples):]

        self.single_sample = single_sample

    def __getitem__(self, idx):
        if self.single_sample:
            idx = 666
        
        label = self.y[idx]
        sample = self.X[idx]

        # preprocess data
        label = action_to_id(label)
        label = torch.from_numpy(label)
        sample = rgb2gray(sample)
        sample = torch.from_numpy(sample)
        
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