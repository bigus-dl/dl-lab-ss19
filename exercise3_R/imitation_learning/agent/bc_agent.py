import sys

import torch
import torch.optim as optim
from imitation_learning.agent.networks import CNN

class BCAgent:
    
    def __init__(self, learning_rate = 1e-4):
        # TODO: Define network, loss function, optimizer
        self.net = CNN(history_length=1,n_classes=5)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

    def update(self, X_batch, y_batch):

        # TODO: forward + backward + optimize
        self.net.train()
        self.optimizer.zero_grad()
        y_hat  = self.net(X_batch)
        print("y_hat {} , y_batch {}".format(y_hat,y_batch))
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
            loss = self.loss_fn(y_hat,y_val)
            return loss.item()

    def load(self, file_name):
        torch.save(self.net.state_dict(), file_name+"_model")
        torch.save(self.optimizer.state_dict(), file_name+"_optimizer")

    def save(self, file_name):
        self.net.load_state_dict(torch.load(file_name+"_model"))
        self.optimizer.load_state_dict(torch.load(file_name+"_optimizer"))
