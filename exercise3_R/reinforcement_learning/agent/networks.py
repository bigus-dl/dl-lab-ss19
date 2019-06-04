import torch.nn as nn
import torch
import torch.nn.functional as F


"""
CartPole network
"""

class MLP(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dim=400):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, action_dim)
    torch.nn.init.xavier_uniform_(self.fc1.weight)
    torch.nn.init.xavier_uniform_(self.fc2.weight)
    torch.nn.init.xavier_uniform_(self.fc3.weight)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.fc3(x)

class CNN(nn.Module):
  def __init__(self,inputs, outputs):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(inputs, 16, kernel_size=6, stride=2)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=6, stride=2)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
    self.bn3 = nn.BatchNorm2d(32)        
  
    linear_input_size = 10 * 10 * 32
    self.dropout = nn.Dropout(0.5)
    self.head = nn.Linear(linear_input_size, 256)
    self.hidden = nn.Linear(256, 16)
    self.tail = nn.Linear(16, outputs)

    torch.nn.init.xavier_uniform_(self.head.weight)
    torch.nn.init.xavier_uniform_(self.hidden.weight)
    torch.nn.init.xavier_uniform_(self.tail.weight)
    
  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))

    x = x.view(x.size(0), -1)
    x = self.dropout(x)
    x = self.head(x)
    x = F.relu(x)
    x = self.hidden(x)
    x = F.relu(x)
    x = self.tail(x)
    return x


