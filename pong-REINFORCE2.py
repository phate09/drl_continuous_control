#!/usr/bin/env python
# coding: utf-8

import argparse
from datetime import datetime

import gym
import matplotlib.pyplot as plt
import numpy as np
import progressbar as pb
import torch
import torch.optim as optim
from scipy import stats
from visdom import Visdom

import pong_utils
from parallelEnv import parallelEnv

device = pong_utils.device
print("using device: ", device)
COST = "Cost"
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
env = gym.make('PongDeterministic-v4')
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
DEFAULT_PORT = 8097
DEFAULT_HOSTNAME = "http://localhost"
parser = argparse.ArgumentParser(description='Demo arguments')
parser.add_argument('-port', metavar='port', type=int, default=DEFAULT_PORT,
                    help='port the visdom server is running on.')
parser.add_argument('-server', metavar='server', type=str,
                    default=DEFAULT_HOSTNAME,
                    help='Server address of the target to run the demo on.')
FLAGS = parser.parse_args()
viz = Visdom(port=FLAGS.port, server=FLAGS.server, env="REINFORCE")
assert viz.check_connection(timeout_seconds=3), 'No connection could be formed quickly, remember to run \'visdom\' in the terminal'

viz.close(win=COST)
policy = pong_utils.Policy().to(device)
# policy = torch.load('REINFORCE2.policy')

optimizer = optim.Adam(policy.parameters(), lr=1e-4)
envs = parallelEnv('PongDeterministic-v4', n=4, seed=12345)
prob, state, action, reward = pong_utils.collect_trajectories(envs, policy, tmax=100)


def surrogate(policy, old_probs, states, actions, rewards,
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


Lsur = surrogate(policy, prob, state, action, reward)

print(Lsur)

# # Training
# WARNING: running through all 800 episodes will take 30-45 minutes
episode = 800
widget = ['training loop: ', pb.Percentage(), ' ',
          pb.Bar(), ' ', pb.ETA()]
timer = pb.ProgressBar(widgets=widget, maxval=episode).start()

# initialize environment
envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)

discount_rate = .99
epsilon = 0.1
beta = .01
tmax = 320

# keep track of progress
mean_rewards = []
viz.line(X=[0], Y=[0], win=COST, name=dt_string, opts=dict(title='Cost plot'), update='new')  # generates a new trace
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

    # the clipping parameter reduces as time goes on
    epsilon *= .999

    # the regulation term also reduces
    # this reduces exploration in later runs
    beta *= .995

    # get the average reward of the parallel environments
    mean_rewards.append(np.mean(total_rewards))
    viz.line(X=list(range(len(mean_rewards))), Y=mean_rewards, win=COST, name=dt_string, opts=dict(title='Cost plot'))
    # display some progress every 20 iterations
    # if (e + 1) % 20 == 0:
    #     print("\nEpisode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)))
    #     print(total_rewards)

    # update progress widget bar
    timer.update(e + 1)

timer.finish()

# play game after training!
pong_utils.play(env, policy, time=2000)
plt.plot(mean_rewards)
# save your policy!
# torch.save(policy, 'REINFORCE3.policy')

# load your policy if needed
# policy = torch.load('REINFORCE.policy')

# try and test out the solution!
# policy = torch.load('PPO_solution.policy')
