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
        self.conv1 = torch.nn.Conv2d(history_length, 32, kernel_size=5, stride=1, padding=1)
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # input 47x47
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1)
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # input 22x22
        self.conv3 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # input 32x11x11
        self.fc1 = torch.nn.Linear(32*11*11, 64)
        self.fc2 = torch.nn.Linear(64, 16)
        self.fc3 = torch.nn.Linear(16, n_classes)
        self.rlu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.rlu(x)
        x = self.conv2(x)
        x = self.rlu(x)
        x = self.conv3(x)
        x = self.rlu(x)
        print(x.shape)
        x = x.view(x.size(0),-1)

        x = self.fc1(x)
        x = self.rlu(x)
        x = self.fc2(x)

        return x

