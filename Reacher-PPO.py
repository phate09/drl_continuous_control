import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment

import constants
from agents.Agent_PPO_continuous import AgentPPO, Policy


def main():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    current_time = now.strftime('%b%d_%H-%M-%S')
    device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device: ", device)
    seed = 5
    np.random.seed(seed)
    env = UnityEnvironment(file_name="environment/Reacher_Linux_NoVis/Reacher.x86_64", seed=seed)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    example_state = env_info.vector_observations[0]
    print('States look like:', example_state)
    state_size = len(example_state)
    print('States have length:', state_size)

    comment = f"PPO Unity"
    policy = Policy(state_size, action_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    ending_condition = lambda result: result['mean'] >= 30.0
    config = {constants.optimiser: optimizer,
              constants.model: policy,
              constants.n_episodes: 2000*8,
              constants.max_t: 2000,
              constants.epsilon: 0.2,
              constants.beta: 0.01,
              constants.input_dim: (1, 1),
              constants.output_dim: (1, 1),
              constants.discount: 0.99,
              constants.device: device,
              constants.sgd_iterations: 6,
              constants.ending_condition: ending_condition
              }
    log_dir = os.path.join('runs', current_time + '_' + comment)
    os.mkdir(log_dir)
    config_file = open(os.path.join(log_dir, "config.json"), "w+")
    # magic happens here to make it pretty-printed
    config_file.write(json.dumps(config, indent=4, sort_keys=True, default=lambda o: '<not serializable>'))
    config_file.close()
    writer = SummaryWriter(log_dir=log_dir)
    agent = AgentPPO(config)
    agent.train(env,brain_name,writer,ending_condition)
    print("Finished.")


if __name__ == '__main__':
    main()
