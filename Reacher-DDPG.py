import json
import os
from datetime import datetime

import jsonpickle
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment

import constants
from agents.Unity.Agent_DDPG import AgentDDPG
from networks.actor_critic.Policy_actor import Policy_actor
# from environment.Reacher_wrapper import Reacher_wrapper
from networks.actor_critic.Policy_critic import Policy_critic


def main():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    current_time = now.strftime('%b%d_%H-%M-%S')
    device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device: ", device)
    seed = 2
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = UnityEnvironment("./environment/Reacher_Linux_NoVis/Reacher.x86_64",worker_id=0, seed=seed,no_graphics=True)
    brain = env.brains[env.brain_names[0]]
    env_info = env.reset(train_mode=True)[env.brain_names[0]]
    print('Number of agents:', len(env_info.agents))
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size
    action_type = brain.vector_action_space_type
    comment = f"DDPG Unity Reacher"
    actor = Policy_actor(state_size, action_size).to(device)
    critic = Policy_critic(state_size + action_size).to(device)
    # actor.test(device)
    optimizer_actor = optim.Adam(actor.parameters(), lr=1e-4)
    optimizer_critic = optim.Adam(critic.parameters(), lr=1e-4)
    # optimizer = optim.RMSprop(actor.parameters(),lr=1e-4)
    ending_condition = lambda result: result['mean'] >= 30.0
    log_dir = os.path.join('runs', current_time + '_' + comment)
    os.mkdir(log_dir)
    config = {
        constants.optimiser_actor: optimizer_actor,
        constants.optimiser_critic: optimizer_critic,
        constants.model_actor: actor,
        constants.model_critic: critic,
        constants.n_episodes: 500,
        constants.batch_size: 256,
        constants.buffer_size: int(1e6),
        constants.max_t: 2000,  # just > 1000
        constants.input_dim: state_size,
        constants.output_dim: action_size,
        constants.gamma: 0.99, #discount
        constants.tau: 0.001, #soft merge
        constants.device: device,
        constants.train_every: 4,
        constants.train_n_times: 1,
        constants.ending_condition: ending_condition,
        constants.log_dir: log_dir
    }
    config_file = open(os.path.join(log_dir, "config.json"), "w+")
    config_file.write(json.dumps(json.loads(jsonpickle.encode(config, unpicklable=False, max_depth=1)), indent=4, sort_keys=True))
    config_file.close()
    writer = SummaryWriter(log_dir=log_dir)
    agent = AgentDDPG(config)
    agent.train(env, writer, ending_condition)
    print("Finished.")


if __name__ == '__main__':
    main()
