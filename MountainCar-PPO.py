import json
import os
from datetime import datetime

import gym
import jsonpickle
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

import constants
from agents.Agent_PPO_OpenAI_continuous import AgentPPO
from networks.Policy_continuous import Policy


def main():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    current_time = now.strftime('%b%d_%H-%M-%S')
    device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device: ", device)
    seed = 5
    np.random.seed(seed)
    env = gym.make("MountainCarContinuous-v0")
    # get the default brain
    # brain_name = env.brain_names[0]
    # brain = env.brains[brain_name]

    # reset the environment
    # env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    # print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = env.action_space.shape
    print('Number of actions:', action_size)

    # examine the state space
    example_state = env.reset()
    print('States look like:', example_state)
    state_size = len(example_state)
    print('States have length:', state_size)

    comment = f"PPO OpenAI mountain car"
    policy = Policy(state_size, action_size[0]).to(device)
    # policy.test(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    # optimizer = optim.RMSprop(policy.parameters(),lr=1e-4)
    ending_condition = lambda result: result['mean'] >= 30.0
    log_dir = os.path.join('runs', current_time + '_' + comment)
    os.mkdir(log_dir)
    config = {constants.optimiser: optimizer,
              constants.model: policy,
              constants.n_episodes: 2000 * 8,
              constants.max_t: 2000,
              constants.epsilon: 0.2,
              constants.beta: 0.01,
              constants.input_dim: (1, 1),
              constants.output_dim: (1, 1),
              constants.discount: 0.99,
              constants.device: device,
              constants.sgd_iterations: 6,
              constants.ending_condition: ending_condition,
              constants.log_dir: log_dir
              }
    config_file = open(os.path.join(log_dir, "config.json"), "w+")
    config_file.write(json.dumps(json.loads(jsonpickle.encode(config, unpicklable=False, max_depth=1)), indent=4, sort_keys=True))
    config_file.close()
    writer = SummaryWriter(log_dir=log_dir)
    agent = AgentPPO(config)
    agent.train(env, None, writer, ending_condition)
    print("Finished.")


if __name__ == '__main__':
    main()


class ConfigJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, optim.Optimizer):
            return f"{obj}"
        else:
            return json.JSONEncoder.default(self, obj)
