#!/usr/bin/env python
# coding: utf-8
import argparse

import gym
import matplotlib.pyplot as plt
import numpy as np
import progressbar as pb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
from visdom import Visdom

import pong_utils
from parallelEnv import parallelEnv

COST = "cost"

device = pong_utils.device
print("using device: ", device)
env = gym.make('PongDeterministic-v4')

print("List of available actions: ", env.unwrapped.get_action_meanings())

# # show what a preprocessed image looks like
# env.reset()
# _, _, _, _ = env.step(0)
# # get a frame after 20 steps
# for _ in range(20):
#     frame, _, _, _ = env.step(1)
#
# plt.subplot(1, 2, 1)
# plt.imshow(frame)
# plt.title('original image')
#
# plt.subplot(1, 2, 2)
# plt.title('preprocessed image')

# 80 x 80 black and white image
# plt.imshow(pong_utils.preprocess_single(frame), cmap='Greys')
# plt.show()
DEFAULT_PORT = 8097
DEFAULT_HOSTNAME = "http://localhost"
parser = argparse.ArgumentParser(description='Demo arguments')
parser.add_argument('-port', metavar='port', type=int, default=DEFAULT_PORT,
                    help='port the visdom server is running on.')
parser.add_argument('-server', metavar='server', type=str,
                    default=DEFAULT_HOSTNAME,
                    help='Server address of the target to run the demo on.')
FLAGS = parser.parse_args()
viz = Visdom(port=FLAGS.port, server=FLAGS.server)
assert viz.check_connection(timeout_seconds=3), 'No connection could be formed quickly, remember to run \'visdom\' in the terminal'

viz.close(win=COST)
# set up a convolutional neural net
# the output is the probability of moving right
# P(left) = 1-P(right)
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
        x = self.max1(F.relu(self.conv1(x)))
        x = self.max2(F.relu(self.conv2(x)))
        x = self.max3(F.relu(self.conv3(x)))
        # flatten the tensor
        x = x.view(-1, self.size)
        return self.sig(self.fc(x))


# run your own policy!
policy=Policy().to(device)
# policy = pong_utils.Policy().to(device)
policy = torch.load('PPO2.policy')
optimizer = optim.Adam(policy.parameters(), lr=1e-4)

pong_utils.play(env, policy, time=200)


def clipped_surrogate(policy, old_probs, states, actions, rewards,
                      discount=0.995, epsilon=0.1, beta=0.01):
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    # discount = discount ** np.arange(len(rewards))
    rewards.reverse()
    previous_rewards = 0
    for i in range(len(rewards)):
        rewards[i] = rewards[i] + discount * previous_rewards
        previous_rewards = rewards[i]
    rewards.reverse()
    # mean = np.mean(rewards, axis=1)
    # std = np.std(rewards, axis=1) + 1.0e-10
    # rewards_normalized = (rewards - mean[:, np.newaxis]) / std[:, np.newaxis]
    rewards_standardised = stats.zscore(rewards, axis=1)
    rewards_standardised = np.nan_to_num(rewards_standardised, False)
    # means = np.mean(rewards, axis=1)
    # stds = np.std(rewards, axis=1) + 1.0e-10
    # rewards_standardised = rewards - means / stds
    assert not np.isnan(rewards_standardised).any()
    rewards_standardised = torch.tensor(rewards_standardised, dtype=torch.float, device=device)

    # convert states to policy (or probability)
    new_probs = pong_utils.states_to_prob(policy, states)
    new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0 - new_probs)

    # cost = torch.log(new_probs) * rewards_standardised
    ratio = new_probs / old_probs
    cost = torch.min(ratio, torch.clamp(ratio, 1 - epsilon, i + epsilon)) * rewards_standardised
    # include a regularization term
    # this steers new_policy towards 0.5
    # which prevents policy to become exactly 0 or 1
    # this helps with exploration
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs * torch.log(old_probs + 1.e-10) + (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))

    my_surrogate = torch.mean(cost + beta * entropy)
    # surrogate = pong_utils.surrogate(policy, old_probs, states, actions, rewards, discount, beta)
    return my_surrogate


# # Training

episode = 1000

widget = ['training loop: ', pb.Percentage(), ' ',
          pb.Bar(), ' ', pb.ETA()]
timer = pb.ProgressBar(widgets=widget, maxval=episode).start()

envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)

discount_rate = .99
epsilon = 0.1
beta = .01
tmax = 320
SGD_epoch = 4

# keep track of progress
mean_rewards = []

for e in range(episode):

    # collect trajectories
    old_probs, states, actions, rewards = pong_utils.collect_trajectories(envs, policy, tmax=tmax)

    total_rewards = np.sum(rewards, axis=0)

    # gradient ascent step
    for _ in range(SGD_epoch):
        # uncomment to utilize your own clipped function!
        L = -clipped_surrogate(policy, old_probs, states, actions, rewards, epsilon=epsilon, beta=beta)

        # L = -pong_utils.clipped_surrogate(policy, old_probs, states, actions, rewards,epsilon=epsilon, beta=beta)
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        del L

    # the clipping parameter reduces as time goes on
    epsilon *= .999

    # the regulation term also reduces
    # this reduces exploration in later runs
    beta *= .995

    # get the average reward of the parallel environments
    mean_rewards.append(np.mean(total_rewards))
    viz.line(X=list(range(len(mean_rewards))), Y=mean_rewards, win=COST, opts=dict(title='Cost plot'))
    # display some progress every 20 iterations
    # if (e + 1) % 20 == 0:
    #     print("Episode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)))
    #     print(total_rewards)

    # update progress widget bar
    timer.update(e + 1)

timer.finish()

pong_utils.play(env, policy, time=200)

# save your policy!
torch.save(policy, 'PPO2.policy')

# load policy if needed
# policy = torch.load('PPO.policy')

# try and test out the solution 
# make sure GPU is enabled, otherwise loading will fail
# (the PPO verion can win more often than not)!
#
# policy_solution = torch.load('PPO_solution.policy')
# pong_utils.play(env, policy_solution, time=2000)
