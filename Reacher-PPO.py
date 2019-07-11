import datetime
import os
from collections import deque

import numpy as np
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment

from agents.RandomAgent import RandomAgent


def main():
    currentDT = datetime.datetime.now()
    print(f'Start at {currentDT.strftime("%Y-%m-%d %H:%M:%S")}')
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

    current_time = currentDT.strftime('%b%d_%H-%M-%S')
    comment = f"random_agent"
    log_dir = os.path.join('runs', current_time + '_' + comment)
    writer = SummaryWriter(log_dir=log_dir)
    agent = RandomAgent(state_size=state_size, action_size=action_size, seed=0)

    def train(n_episodes=2000, max_t=1000):
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores

        for i_episode in range(n_episodes):
            env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
            agent.reset()  # reset the agent
            state = env_info.vector_observations[0]  # get the current state
            score = 0
            for t in range(max_t):
                action = agent.act(state)
                env_info = env.step(action)[brain_name]  # send the action to the environment
                next_state = env_info.vector_observations[0]  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished
                agent.collect(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            agent.learn()
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            writer.add_scalar('data/score', score, i_episode)
            writer.add_scalar('data/score_average', np.mean(scores_window), i_episode)
            # writer.add_scalar('data/epsilon', eps.get(i_episode), i_episode)
            # writer.add_scalar('data/beta', betas.get(i_episode), i_episode)
            # eps = max(eps_end, eps_decay * eps)  # decrease epsilon
            print(
                f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} ', end="")
            if i_episode + 1 % 100 == 0:
                print(f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} ')
            # torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            if np.mean(scores_window) >= 30.0:
                print(f'\nEnvironment solved in {i_episode - 100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
                # torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth') #save the agent
                break
        return scores

    train()
    print("Finished.")


if __name__ == '__main__':
    main()
