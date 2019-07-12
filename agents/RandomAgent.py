import random

import numpy as np

import constants
from agents.GenericAgent import GenericAgent


class RandomAgent(GenericAgent):
    """Interacts with and learns from the environment."""

    def __init__(self, config):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super().__init__(config)
        self.input_dim = config[constants.input_dim]
        self.output_dim = config[constants.output_dim]
        self.seed = random.seed(config[constants.seed])

    def collect(self, state, action, reward, next_state, done):
        pass

    def act(self, state=None):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        return np.random.rand(self.action_size)

    def learn(self):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        pass

    def required_properties(self):
        return [constants.input_dim, constants.output_dim, constants.seed]

    def reset(self):
        pass


if __name__ == '__main__':
    agent = RandomAgent({constants.input_dim: 1,
                         constants.output_dim: 1,
                         constants.seed: 0})
    print('Test passed')
