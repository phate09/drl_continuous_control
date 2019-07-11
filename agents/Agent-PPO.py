import random
from collections import deque

import numpy as np
import torch
from scipy import stats


class AgentPPO():
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
        self.state_list = []
        self.action_list = []
        self.prob_list = []
        self.reward_list = []

    # self.t_update_target_step = 0

    def collect(self, state, action, probs, reward):
        self.state_list.append(state)
        self.action_list.append(action)
        self.prob_list.append(probs)
        self.reward_list.append(reward)
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

    def reset(self):
        """Reset the memory of the agent"""
        self.state_list = []
        self.action_list = []
        self.prob_list = []
        self.reward_list = []

    def train(self, env, brain_name, writer, ending_condition, n_episodes=2000, max_t=1000):
        """

        :param env:
        :param brain_name:
        :param writer:
        :param ending_condition: a method that given a score window returns true or false
        :param n_episodes:
        :param max_t:
        :return:
        """
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores

        for i_episode in range(n_episodes):
            env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
            self.reset()  # reset the agent
            state = env_info.vector_observations[0]  # get the current state
            score = 0
            for t in range(max_t):
                action, probs = self.act(state)
                env_info = env.step(action)[brain_name]  # send the action to the environment
                next_state = env_info.vector_observations[0]  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished
                self.collect(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            self.learn()
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            writer.add_scalar('data/score', score, i_episode)
            writer.add_scalar('data/score_average', np.mean(scores_window), i_episode)
            print(
                f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} ', end="")
            if i_episode + 1 % 100 == 0:
                print(f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} ')
            # torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            if ending_condition(scores_window):
                print(f'\nEnvironment solved in {i_episode - 100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
                # torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth') #save the agent
                break
        return scores


# def clipped_surrogate(policy, old_probs, states, actions, rewards,
#                       discount=0.995, epsilon=0.1, beta=0.01):
#     actions = torch.tensor(actions, dtype=torch.int8, device=device)
#     old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
#     rewards.reverse()
#     previous_rewards = 0
#     for i in range(len(rewards)):
#         rewards[i] = rewards[i] + discount * previous_rewards
#         previous_rewards = rewards[i]
#     rewards.reverse()
#     rewards_standardised = stats.zscore(rewards, axis=1)
#     rewards_standardised = np.nan_to_num(rewards_standardised, False)
#     assert not np.isnan(rewards_standardised).any()
#     rewards_standardised = torch.tensor(rewards_standardised, dtype=torch.float, device=device)
#
#     # convert states to policy (or probability)
#     new_probs = pong_utils.states_to_prob(policy, states)
#     new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0 - new_probs)
#
#     # cost = torch.log(new_probs) * rewards_standardised
#     ratio = new_probs / old_probs
#     cost = torch.min(ratio, torch.clamp(ratio, 1 - epsilon, i + epsilon)) * rewards_standardised
#     # include a regularization term
#     # this steers new_policy towards 0.5
#     # which prevents policy to become exactly 0 or 1
#     # this helps with exploration
#     # add in 1.e-10 to avoid log(0) which gives nan
#     entropy = -(new_probs * torch.log(old_probs + 1.e-10) + (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))
#
#     my_surrogate = torch.mean(cost + beta * entropy)
#     return my_surrogate


# def collect_trajectories(envs, agent, tmax=200):
#     # number of parallel instances
#     n = len(envs.ps)
#
#     # initialize returning lists and start the game!
#     state_list = []
#     reward_list = []
#     prob_list = []
#     action_list = []
#
#     envs.reset()
#
#     # start all parallel agents
#     envs.step([1] * n)
#
#     # perform nrand random steps
#     for _ in range(nrand):
#         fr1, re1, _, _ = envs.step(np.random.choice([RIGHT, LEFT], n))
#         fr2, re2, _, _ = envs.step([0] * n)
#
#     for t in range(tmax):
#
#         # prepare the input
#         # preprocess_batch properly converts two frames into
#         # shape (n, 2, 80, 80), the proper input for the policy
#         # this is required when building CNN with pytorch
#         batch_input = preprocess_batch([fr1, fr2])
#
#         # probs will only be used as the pi_old
#         # no gradient propagation is needed
#         # so we move it to the cpu
#         probs = policy(batch_input).squeeze().cpu().detach().numpy()
#
#         action = np.where(np.random.rand(n) < probs, RIGHT, LEFT)
#         probs = np.where(action == RIGHT, probs, 1.0 - probs)
#
#         # advance the game (0=no action)
#         # we take one action and skip game forward
#         fr1, re1, is_done, _ = envs.step(action)
#         fr2, re2, is_done, _ = envs.step([0] * n)
#
#         reward = re1 + re2
#
#         # store the result
#         state_list.append(batch_input)
#         reward_list.append(reward)
#         prob_list.append(probs)
#         action_list.append(action)
#
#         # stop if any of the trajectories is done
#         # we want all the lists to be rectangular
#         if is_done.any():
#             break

    # # return pi_theta, states, actions, rewards, probability
    # return prob_list, state_list, \
    #        action_list, reward_list
