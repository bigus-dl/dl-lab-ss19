import sys

import torch
import torch.optim as optim
from imitation_learning.agent.networks import CNN

class BCAgent:
    
    def __init__(self, learning_rate = 1e-4, cuda = None, weighted=False, history=1):
        # TODO: Define network, loss function, optimizer
        '''
                    left            right           accel.          brake           straight
        count :     8784            4038            13639           1118            22421
        median: 
        weight:     1.              2.17533432      0.64403549      7.8568873       0.39177557
        normal:     0.17568         0.08076         0.27278         0.02236         0.44842
        revers:     5.692167577     12.382367508    3.665957914     44.722719141    2.230052183
        normal:     0.083708346     0.1820936398    0.053911145     0.6576870461    0.032794885
        sftmax:     0.1724          0.1902          0.1674          0.3061          0.1639
        sftmax:     0.1943          0.1978          0.1933          0.2221          0.1926
        '''
        self.net = CNN(history_length=history,n_classes=5)
        self.class_weights = torch.Tensor([1, 1, 1, 1, 1]).to(cuda)
        if weighted:
            self.class_weights = torch.Tensor([1., 2.17533432, 0.64403549, 7.8568873, 0.39177557]).to(cuda)
        self.net.to(cuda)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

    def update(self, X_batch, y_batch):

        # TODO: forward + backward + optimizer
        self.net.train()
        self.optimizer.zero_grad()
        y_hat  = self.net(X_batch)
        loss = self.loss_fn(y_hat, y_batch.squeeze())
        loss.backward()
        self.optimizer.step()

        loss = loss.item()
        return loss
    
    @torch.no_grad()
    def predict(self, X):
        self.net.eval()
        output = self.net(X)
        return output

    @torch.no_grad()
    def validate(self, X_val,y_val):
        self.net.eval()
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
