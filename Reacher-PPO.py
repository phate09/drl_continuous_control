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
from agents.Agent_PPO_continuous import AgentPPO
from networks.actor_critic.actorPPO import Policy_actor
# from environment.Reacher_wrapper import Reacher_wrapper


def main():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    current_time = now.strftime('%b%d_%H-%M-%S')
    device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device: ", device)
    seed = 5
    np.random.seed(seed)
    env = UnityEnvironment("./environment/Reacher_Linux_NoVis/Reacher.x86_64",seed=seed)
    brain = env.brains[env.brain_names[0]]
    env_info = env.reset(train_mode=True)[env.brain_names[0]]
    print('Number of agents:', len(env_info.agents))
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size
    action_type = brain.vector_action_space_type
    comment = f"PPO Unity Reacher"
    policy = Policy_actor(state_size, action_size).to(device)
    # policy.test(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    # optimizer = optim.RMSprop(policy.parameters(),lr=1e-4)
    ending_condition = lambda result: result['mean'] >= 30.0
    log_dir = os.path.join('runs', current_time + '_' + comment)
    os.mkdir(log_dir)
    config = {constants.optimiser: optimizer,
              constants.model: policy,
              constants.n_episodes: 2000 * 8,
              constants.max_t: 1000,
              constants.epsilon: 0.2,
              constants.beta: 0.01,
              constants.input_dim: state_size,
              constants.output_dim: action_size,
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
    agent.train(env, writer, ending_condition)
    print("Finished.")


if __name__ == '__main__':
    main()