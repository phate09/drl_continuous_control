from torch import nn as nn
from torch.nn import functional as F


class DummyNetwork(nn.Module):
    def __init__(self, input_dim=26, output_dim=2):
        super(DummyNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, output_dim)
        # Sigmoid to
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x
