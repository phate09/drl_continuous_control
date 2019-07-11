import random

import numpy as np
import torch


class RandomAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    # self.t_update_target_step = 0

    def step(self, state, action, reward, next_state, done):
        pass

    def act(self, state=None):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        return np.random.rand(self.action_size)

    def learn(self, experiences, indexes, is_values):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        pass
def clipped_surrogate(policy, old_probs, states, actions, rewards,
                      discount=0.995, epsilon=0.1, beta=0.01):
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards.reverse()
    previous_rewards = 0
    for i in range(len(rewards)):
        rewards[i] = rewards[i] + discount * previous_rewards
        previous_rewards = rewards[i]
    rewards.reverse()
    rewards_standardised = stats.zscore(rewards, axis=1)
    rewards_standardised = np.nan_to_num(rewards_standardised, False)
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
    return my_surrogate