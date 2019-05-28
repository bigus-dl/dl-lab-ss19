import sys

sys.path.append("../")
import torch._C as torch
import torch.optim as optim
from exercise3_R.imitation_learning.agent.networks import CNN

class BCAgent:
    
    def __init__(self, learning_rate = 1e-4):
        # TODO: Define network, loss function, optimizer
        self.net = CNN(history_length=1,n_classes=4)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.cuda = torch.device('cuda')
        pass
    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        x = torch.from_numpy(X_batch).to(self.cuda)
        y = torch.from_numpy(y_batch).to(self.cuda)

        # TODO: forward + backward + optimize
        self.net.train()
        self.optimizer.zero_grad()
        y_hat  = self.net(x)
        loss = self.loss_fn(y_hat,y)
        loss.backward()
        self.optimizer.step()

        loss = torch.mean(loss).item()
        return loss

    def predict(self, X):
        self.net.eval()
        return self.net(X)
    def load(self, file_name):
        torch.save(self.net.state_dict(), file_name+"_model")
        torch.save(self.optimizer.state_dict(), file_name+"_optimizer")

    def save(self, file_name):
        self.net.load_state_dict(torch.load(file_name+"_model"))
        self.optimizer.load_state_dict(torch.load(file_name+"_optimizer"))
