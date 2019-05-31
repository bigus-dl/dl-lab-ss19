import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""

class CNN(nn.Module):

    def __init__(self, history_length=0, n_classes=5): 
        super(CNN, self).__init__()

        # input 96x96
        self.conv1 = torch.nn.Conv2d(history_length, 16, kernel_size=4, stride=2, padding=0)

        # input 48x48
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0)

        # input 24x24
        self.fc1 = torch.nn.Linear(32*24*24, 32)
        self.rl1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, n_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        #print("before ", x.shape)
        x = x.view(x.size(0),-1)
        #print("after ", x.shape)

        x = self.fc1(x)
        x = self.rl1(x)
        x = self.fc2(x)

        return x

