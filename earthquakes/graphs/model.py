import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    """Simple neural network with 3 dense layers and an output with a sigmoid activation function"""

    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
