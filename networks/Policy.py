import torch
import torchtest
from torch import nn as nn
from torch.nn import functional as F


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        # 80x80x2 to 38x38x4
        # 2 channel from the stacked frame
        self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=1, bias=False)
        self.max1 = nn.MaxPool2d(2)
        # 38x38x4 to 9x9x32
        self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=2)
        self.max2 = nn.MaxPool2d(2)
        self.size = 8 * 8 * 16

        # two fully connected layer
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)

        # Sigmoid to
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.max1(F.relu(self.conv1(x)))
        x = self.max2(F.relu(self.conv2(x)))

        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        return self.sig(self.fc2(x))


class Policy2(nn.Module):

    def __init__(self):
        super(Policy2, self).__init__()
        # 80x80x2 to 38x38x4
        # 2 channel from the stacked frame
        self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=1)
        self.max1 = nn.MaxPool2d(2)
        # 38x38x4 to 9x9x32
        self.conv2 = nn.Conv2d(4, 2, kernel_size=3, stride=1)
        self.max2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(2, 1, kernel_size=3, stride=1)
        self.max3 = nn.MaxPool2d(2)
        self.size = 7 * 7

        # two fully connected layer
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)

        # Sigmoid to
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.max1(F.relu(self.conv1(x)))
        x = self.max2(F.relu(self.conv2(x)))
        x = self.max3(F.relu(self.conv3(x)))

        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        return self.sig(self.fc2(x))

    def test(self, device='cpu'):
        input = torch.randn(1, 2, 80, 80, requires_grad=False)
        targets = torch.rand(1, 1, requires_grad=False)
        torchtest.test_suite(self, F.binary_cross_entropy, torch.optim.Adam(self.parameters()), batch=[input, targets], test_vars_change=True, test_inf_vals=True, test_nan_vals=True, device=device)
        print('All tests passed')


if __name__ == '__main__':
    model = Policy2()
    input = torch.randn(1, 2, 80, 80, requires_grad=False)
    targets = torch.rand(1, 1, requires_grad=False)
    torchtest.test_suite(model, F.binary_cross_entropy, torch.optim.Adam(model.parameters()), batch=[input, targets], test_vars_change=True, test_inf_vals=True, test_nan_vals=True)
    print('All tests passed')
