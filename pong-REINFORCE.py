#!/usr/bin/env python
# coding: utf-8

# # Welcome!
# Below, we will learn to implement and train a policy to play atari-pong, using only the pixels as input. We will use convolutional neural nets, multiprocessing, and pytorch to implement and train our policy. Let's get started!
#
# (I strongly recommend you to try this notebook on the Udacity workspace first before running it locally on your desktop/laptop, as performance might suffer in different environments)

# In[5]:


# custom utilies for displaying animation, collecting rollouts and more
import pong_utils
from scipy import stats

# check which device is being used.
# I recommend disabling gpu until you've made sure that the code runs
device = pong_utils.device
print("using device: ", device)

# In[2]:


# render ai gym environment
import gym
import time

# PongDeterministic does not contain random frameskip
# so is faster to train than the vanilla Pong-v4 environment
env = gym.make('PongDeterministic-v4')

# print("List of available actions: ", env.unwrapped.get_action_meanings())

# we will only use the actions 'RIGHTFIRE' = 4 and 'LEFTFIRE" = 5
# the 'FIRE' part ensures that the game starts again after losing a life
# the actions are hard-coded in pong_utils.py


# # Preprocessing
# To speed up training, we can simplify the input by cropping the images and use every other pixel
#
#

# In[3]:


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        x = F.relu(self.max1(self.conv1(x)))
        x = F.relu(self.max2(self.conv2(x)))
        x = F.relu(self.max3(self.conv3(x)))
        # flatten the tensor
        x = x.view(-1, self.size)
        return self.sig(self.fc(x))


# use your own policy!
# policy = Policy().to(device)
policy = pong_utils.Policy().to(device)

# we use the adam optimizer with learning rate 2e-4
# optim.SGD is also possible
import torch.optim as optim

optimizer = optim.Adam(policy.parameters(), lr=1e-4)

# # Game visualization
# pong_utils contain a play function given the environment and a policy. An optional preprocess function can be supplied. Here we define a function that plays a game and shows learning progress

# In[10]:


# pong_utils.play(env, policy, time=100, preprocess=pong_utils.preprocess_single)
# try to add the option "preprocess=pong_utils.preprocess_single"
# to see what the agent sees


# # Rollout
# Before we start the training, we need to collect samples. To make things efficient we use parallelized environments to collect multiple examples at once

# In[ ]:


envs = pong_utils.parallelEnv('PongDeterministic-v4', n=4, seed=12345)
prob, state, action, reward = pong_utils.collect_trajectories(envs, policy, tmax=100)

# In[ ]:


print(reward)


# # Function Definitions
# Here you will define key functions for training.
#
# ## Exercise 2: write your own function for training
# (this is the same as policy_loss except the negative sign)
#
# ### REINFORCE
# you have two choices (usually it's useful to divide by the time since we've normalized our rewards and the time of each trajectory is fixed)
#
# 1. $\frac{1}{T}\sum^T_t R_{t}^{\rm future}\log(\pi_{\theta'}(a_t|s_t))$
# 2. $\frac{1}{T}\sum^T_t R_{t}^{\rm future}\frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}$ where $\theta'=\theta$ and make sure that the no_grad is enabled when performing the division

# In[13]:


def surrogate(policy, old_probs, states, actions, rewards,
              discount=0.995, beta=0.01):
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
    rewards_standardised = np.nan_to_num(rewards_standardised,False)
    # means = np.mean(rewards, axis=1)
    # stds = np.std(rewards, axis=1) + 1.0e-10
    # rewards_standardised = rewards - means / stds
    assert not np.isnan(rewards_standardised).any()
    rewards_standardised = torch.tensor(rewards_standardised, dtype=torch.float, device=device)

    # convert states to policy (or probability)
    new_probs = pong_utils.states_to_prob(policy, states)
    new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0 - new_probs)

    # cost = torch.log(new_probs) * rewards_standardised
    cost = new_probs / old_probs * rewards_standardised
    # include a regularization term
    # this steers new_policy towards 0.5
    # which prevents policy to become exactly 0 or 1
    # this helps with exploration
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs * torch.log(old_probs + 1.e-10) + (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))

    my_surrogate = torch.mean(cost + beta * entropy)
    # surrogate = pong_utils.surrogate(policy, old_probs, states, actions, rewards, discount, beta)
    return my_surrogate


Lsur = surrogate(policy, prob, state, action, reward)

print(Lsur)

# # Training
# We are now ready to train our policy!
# WARNING: make sure to turn on GPU, which also enables multicore processing. It may take up to 45 minutes even with GPU enabled, otherwise it will take much longer!

# In[ ]:


from parallelEnv import parallelEnv
import numpy as np

# WARNING: running through all 800 episodes will take 30-45 minutes

# training loop max iterations
# episode = 500
episode = 800

# widget bar to display progress
import progressbar as pb

widget = ['training loop: ', pb.Percentage(), ' ',
          pb.Bar(), ' ', pb.ETA()]
timer = pb.ProgressBar(widgets=widget, maxval=episode).start()

# initialize environment
envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)

discount_rate = .99
beta = .01
tmax = 320

# keep track of progress
mean_rewards = []

for e in range(episode):

    # collect trajectories
    old_probs, states, actions, rewards = pong_utils.collect_trajectories(envs, policy, tmax=tmax)

    total_rewards = np.sum(rewards, axis=0)

    # this is the SOLUTION!
    # use your own surrogate function
    L = -surrogate(policy, old_probs, states, actions, rewards, beta=beta)

    # L = -pong_utils.surrogate(policy, old_probs, states, actions, rewards, beta=beta)
    optimizer.zero_grad()
    L.backward()
    optimizer.step()
    del L

    # the regulation term also reduces
    # this reduces exploration in later runs
    beta *= .995

    # get the average reward of the parallel environments
    mean_rewards.append(np.mean(total_rewards))

    # display some progress every 20 iterations
    if (e + 1) % 20 == 0:
        print("Episode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)))
        print(total_rewards)

    # update progress widget bar
    timer.update(e + 1)

timer.finish()

# In[15]:


# play game after training!
pong_utils.play(env, policy, time=2000)

# In[ ]:


plt.plot(mean_rewards)

# In[ ]:


# save your policy!
torch.save(policy, 'REINFORCE.policy')

# load your policy if needed
# policy = torch.load('REINFORCE.policy')

# try and test out the solution!
# policy = torch.load('PPO_solution.policy')
