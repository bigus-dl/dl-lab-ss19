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
    def __init__(self, datasets_dir, frac=0.1, single_sample=False, is_train=True, history=1):
        
        print("decompressing data...")

        data_file = os.path.join(datasets_dir, PKG_NAME)
  
        f = gzip.open(data_file,'rb')
        data = pickle.load(f)

        self.is_train = is_train
        self.history = history

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
        self.single_sample = single_sample

    def __getitem__(self, idx):
        if self.single_sample:
            idx = 666

        label = self.y[idx,:]
        samples = torch.zeros(self.history, self.X.shape[1], self.X.shape[2])
        for i in range(self.history) :
            # if the index allows, add previous frames
            if idx-i>=0 :
                sample = self.X[idx-i,:]
                sample = rgb2gray(sample)
                sample = sample/255
                samples[i] = torch.from_numpy(sample)

            # preprocess data
        label = action_to_id(label)
        label = torch.LongTensor([label])

        return label, samples
    def __len__(self):
        return len(self.y)


def get_data_loader(datasets_dir, frac=0.1, batch_size=1, is_train=False, single_sample=False, history=1):

    reader = PickleReader(datasets_dir, frac, single_sample=single_sample, is_train=is_train, history=history)
    
    data_loader = torch.utils.data.DataLoader(reader,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4 if is_train else 1)
    return data_loader