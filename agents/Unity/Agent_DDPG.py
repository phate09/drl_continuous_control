from collections import deque

import numpy as np
import torch
import torch.nn as nn

import constants
from agents.GenericAgent import GenericAgent


class AgentPPO(GenericAgent):
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
        self.actor: torch.nn.Module = config[constants.model_actor]
        self.critic: torch.nn.Module = config[constants.model_critic]
        self.optimiser_actor: torch.optim.Optimizer = config[constants.optimiser]
        self.optimiser_critic: torch.optim.Optimizer = config[constants.optimiser]
        self.ending_condition = config[constants.ending_condition]
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
                constants.model,
                constants.optimiser,
                constants.ending_condition]

    def learn(self, state_list, prob_list, reward_list, epsilon, beta):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        old_log_prob = torch.stack(prob_list, dim=0)
        states = torch.stack(state_list)
        rewards = reward_list  # torch.cat(reward_list)
        self.model.train()
        for s in range(self.sgd_iterations):
            L = self.clipped_surrogate(old_log_prob, states, rewards, epsilon=epsilon, beta=beta)
            self.optimiser.zero_grad()
            L.sum().backward(retain_graph=True if s != self.sgd_iterations - 1 else False)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)  # clips the gradients for stability
            self.optimiser.step()
        self.model.eval()

    def reset(self):
        """Reset the memory of the agent"""
        self.state_list = []
        self.action_list = []
        self.prob_list = []
        self.reward_list = []

    def train(self, env, writer, ending_condition):
        """

        :param env:
        :param brain_name:
        :param writer:
        :param ending_condition: a method that given a score window returns true or false
        :param n_episodes:
        :param max_t:
        :return:
        """
        brain_name = env.brain_names[0]
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        epsilon = self.epsilon
        beta = self.beta
        for i_episode in range(self.n_episodes):
            env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
            # Reset the memory of the agent
            state_list = []
            action_list = []
            prob_list = []
            reward_list = []
            state = torch.tensor(env_info.vector_observations[0], dtype=torch.float, device=self.device)  # get the current state
            score = 0
            self.model.eval()
            for t in range(self.max_t):
                action, log_prob, entropy = self.model(state.unsqueeze(dim=0))
                env_info = env.step(action.cpu().numpy())[brain_name]  # send the action to the environment
                next_state = torch.tensor(env_info.vector_observations[0], dtype=torch.float, device=self.device)  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished
                state_list.append(state)
                action_list.append(action)
                prob_list.append(log_prob)
                reward_list.append(reward)
                state = next_state
                score += reward
                if done:
                    break
            # prepares the rewards
            rewards = reward_list.copy()
            rewards.reverse()
            previous_rewards = 0
            for i in range(len(rewards)):
                rewards[i] = rewards[i] + self.discount * previous_rewards
                previous_rewards = rewards[i]
            rewards.reverse()
            rewards_array = np.asanyarray(rewards)
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
                print(f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} ')
            # torch.save(self.model, os.path.join(log_dir, "checkpoint.pth"))
            result = {"mean": np.mean(scores_window)}
            if ending_condition(result):
                print(f'\nEnvironment solved in {i_episode - 100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
                # torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth') #save the agent
                break
        return scores

    def clipped_surrogate(self, old_log_probs, states, rewards_standardised, discount=0.995, epsilon=0.1, beta=0.01):
        # convert states to policy (or probability)

        actions, log_prob, entropy = self.model(states)
        # cost = torch.log(new_probs) * rewards_standardised
        ratio = torch.exp(log_prob - old_log_probs)
        cost = torch.min(ratio, torch.clamp(ratio, 1 - epsilon, 1 + epsilon))
        my_surrogate = -torch.mean(cost) * rewards_standardised  # + beta * entropy
        return my_surrogate
