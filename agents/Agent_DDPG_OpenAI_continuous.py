import os
from collections import deque

import numpy as np
import torch
import torch.distributions
import torch.nn as nn

import constants
from agents.GenericAgent import GenericAgent


class AgentDDPG(GenericAgent):
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
        self.input_dim: int = config[constants.input_dim]
        self.output_dim: int = config[constants.output_dim]
        # self.seed: int = random.seed(config[constants.seed])
        self.max_t: int = config[constants.max_t]
        self.sgd_iterations: int = config[constants.sgd_iterations]
        self.n_episodes: int = config[constants.n_episodes]
        self.discount: float = config[constants.discount]
        self.epsilon: float = config[constants.epsilon]
        self.beta: float = config[constants.beta]
        self.device = config[constants.device]
        self.model_actor: torch.nn.Module = config[constants.model_actor]
        self.model_critic: torch.nn.Module = config[constants.model_critic]
        self.optimiser: torch.optim.Optimizer = config[constants.optimiser]
        self.ending_condition = config[constants.ending_condition]
        self.log_dir = config[constants.log_dir]
        self.n_games = 1

    # self.t_update_target_step = 0
    def required_properties(self):
        return [constants.input_dim,
                constants.output_dim,
                constants.max_t,
                constants.sgd_iterations,
                constants.n_episodes,
                constants.discount,
                constants.beta,
                constants.device,
                constants.model_actor,
                constants.model_critic,
                constants.optimiser,
                constants.ending_condition]

    def act(self, state=None):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        return np.random.rand(self.action_size)

    def learn(self, state_list, prob_list, reward_list, epsilon, beta):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        self.model_actor.train()
        for s in range(self.sgd_iterations):
            L = -self.clipped_surrogate(prob_list, state_list, reward_list, epsilon=epsilon, beta=beta)
            self.optimiser.zero_grad()
            L.backward(retain_graph=True if s < self.sgd_iterations - 1 else False)
            self.optimiser.step()

    def reset(self):
        """Reset the memory of the agent"""
        self.state_list = []
        self.action_list = []
        self.prob_list = []
        self.reward_list = []

    def train(self, envs, brain_name, writer, ending_condition):
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
        epsilon = self.epsilon
        beta = self.beta
        for i_episode in range(self.n_episodes):
            state = [env.reset() for env in envs]  # reset the environment
            # Reset the memory of the agent
            state_list = []
            action_list = []
            prob_list = []
            reward_list = []
            # state = env_info.vector_observations[0]  # get the current state
            score = 0
            self.model_actor.eval()
            for t in range(self.max_t):
                state = torch.tensor(state, dtype=torch.float, device=self.device)
                action, log_prob, entropy = self.model_actor(state)
                next_state, reward, done, _ = zip(*[envs[i].step(action[i].cpu().numpy()) for i in range(len(envs))])  # send the action to the environment
                reward = list(reward)
                state_list.append(state)
                action_list.append(action)
                prob_list.append(log_prob)
                reward_list.append(reward)
                state = list(next_state)
                score += np.mean(reward)
                if any(done):
                    break
            # prepares the rewards
            rewards = reward_list.copy()
            rewards.reverse()
            rewards = np.asarray(rewards)
            previous_rewards = np.zeros(len(envs))
            for i in range(len(rewards)):
                rewards[i] = rewards[i] + self.discount * previous_rewards
                previous_rewards = rewards[i]
            # rewards.reverse()
            rewards_array = np.flip(rewards, 0)  # np.asanyarray(rewards)
            mns = rewards_array.mean()
            sstd = rewards_array.std() + 1e-6
            rewards_standardised = (rewards_array - mns) / sstd
            rewards_standardised = np.nan_to_num(rewards_standardised, False)
            assert not np.isnan(rewards_standardised).any()
            rewards_standardised = torch.tensor(rewards_standardised, dtype=torch.float, device=self.device)
            reward_list = rewards_standardised
            self.learn(state_list, prob_list, reward_list, epsilon=epsilon, beta=beta)
            # the clipping parameter reduces as time goes on
            epsilon *= .999

            # the regulation term also reduces
            # this reduces exploration in later runs
            beta *= .995
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            writer.add_scalar('data/score', score, i_episode)
            writer.add_scalar('data/score_average', np.mean(scores_window), i_episode)
            print(
                f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} ', end="")
            if i_episode + 1 % 100 == 0:
                print(f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} epsilon: {epsilon:.5f} beta: {beta:.5f}')
                torch.save(self.model_actor, os.path.join(self.log_dir, f"checkpoint_actor_{i_episode + 1}.pth"))
            result = {"mean": np.mean(scores_window)}
            if ending_condition(result):
                print(f'\nEnvironment solved in {i_episode - 100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
                torch.save(self.model_actor, os.path.join(self.log_dir, f"complete_actor_{i_episode + 1}.pth"))
                # torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth') #save the agent
                break
        return scores

    def clipped_surrogate(self, old_log_probs, states, rewards_standardised, discount=0.995, epsilon=0.1, beta=0.01):
        # convert states to policy (or probability)
        states = torch.stack(states)
        states = states.view(-1, states.size()[2])
        actions, log_prob, entropy = self.model_actor(states)
        old_log_probs = torch.stack(old_log_probs)
        old_log_probs = old_log_probs.view(-1, old_log_probs.size()[2])
        rewards_standardised = rewards_standardised.view(-1).unsqueeze(1)
        # cost = torch.log(new_probs) * rewards_standardised
        ratio = torch.exp(log_prob - old_log_probs)
        ratio_clamped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        advantage = torch.min(ratio, ratio_clamped) * rewards_standardised
        my_surrogate = torch.mean(advantage + beta * entropy)
        return my_surrogate


