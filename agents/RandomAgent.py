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
        self.state_size = config[constants.INPUT_DIM]
        self.action_size = config[constants.OUTPUT_DIM]
        self.seed = random.seed(config[constants.SEED])

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
        return [constants.INPUT_DIM, constants.OUTPUT_DIM, constants.SEED]

    def reset(self):
        pass


if __name__ == '__main__':
    agent = RandomAgent({constants.INPUT_DIM: 1,
                         constants.OUTPUT_DIM: 1,
                         constants.SEED: 0})
    print('Test passed')
