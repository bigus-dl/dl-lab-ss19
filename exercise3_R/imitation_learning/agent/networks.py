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
        self.conv1 = torch.nn.Conv2d(history_length, 64, kernel_size=6, stride=2, padding=0)
        self.rlu = torch.nn.ReLU()
        # input 46x46
        self.conv2 = torch.nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=0)
        # input 22x22
        self.fc1 = torch.nn.Linear(32*22*22, 32)
        self.fc2 = torch.nn.Linear(32, n_classes)

        # activation, regul.
        self.rlu = torch.nn.ReLU()
        self.drp = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.rlu(x)

        x = self.conv2(x)
        x = self.rlu(x)

        x = x.view(x.size(0),-1)

        x = self.fc1(x)
        x = self.rlu(x)
        x = self.drp(x)
        
        x = self.fc2(x)
        return x

