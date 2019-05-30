import sys

import torch
import torch.optim as optim
from imitation_learning.agent.networks import CNN

class BCAgent:
    
    def __init__(self, learning_rate = 1e-4):
        # TODO: Define network, loss function, optimizer
        self.net = CNN(history_length=1,n_classes=5)
        '''
        counts and weights
        right       :   4038    : 0,08076, reverse : 12,382367509
        break       :   1118    : 0,02236, reverse : 44,722719141
        left        :   8784    : 0,17568, reverse : 5,692167577
        straight    :   22421   : 0,44842, reverse : 2,230052183
        accelerate  :   13639   : 0,27278, reverse : 3,665957915
        sum         :   50000   : 1      , reverse : 1
        '''
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor([12,382367509, 44,722719141, 5,692167577, 2,230052183, 3,665957915]))
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

    def update(self, X_batch, y_batch):

        # TODO: forward + backward + optimize
        self.net.train()
        self.optimizer.zero_grad()
        y_hat  = self.net(X_batch)
        loss = self.loss_fn(y_hat, y_batch.squeeze())
        loss.backward()
        self.optimizer.step()

        loss = loss.item()
        return loss

    def predict(self, X):
        self.net.eval()
        output = self.net(X)
        return output

    def validate(self, X_val,y_val):
        with torch.no_grad():
            y_hat = self.net(X_val)
            loss = self.loss_fn(y_hat, y_val.squeeze())
            return loss.item()

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name+"_model")
        torch.save(self.optimizer.state_dict(), file_name+"_optimizer")

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name+"_model"))
        self.optimizer.load_state_dict(torch.load(file_name+"_optimizer"))
