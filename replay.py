import pong_utils
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        # 80x80 to outputsize x outputsize
        # outputsize = (inputsize - kernel_size + stride)/stride
        # (round up if not an integer)
        self.conv1 = nn.Conv2d(2, 1, kernel_size=2, stride=1)
        # size (80-2)/1 +1 = 79
        self.max1 = nn.MaxPool2d(2)
        # size 39
        self.conv2 = nn.Conv2d(1, 1, kernel_size=2, stride=1)
        self.max2 = nn.MaxPool2d(2)
        # size 19
        # output = 20x20 here
        self.conv3 = nn.Conv2d(1, 1, kernel_size=2, stride=1)
        self.max3 = nn.MaxPool2d(2)
        # size 9
        self.size = 1 * 9 * 9

        # 1 fully connected layer
        self.fc = nn.Linear(self.size, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.max1(self.conv1(x)))
        x = F.relu(self.max2(self.conv2(x)))
        x = F.relu(self.max3(self.conv3(x)))
        # flatten the tensor
        x = x.view(-1, self.size)
        return self.sig(self.fc(x))


policy = torch.load('REINFORCE.policy')
env = gym.make('PongDeterministic-v4')
pong_utils.play(env, policy, time=2000)
