import pickle
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim.optimizer

import constants
from agents.GenericAgent import GenericAgent
from utility.PrioReplayBuffer import PrioReplayBuffer


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
        self.gamma: float = config[constants.gamma]
        self.epsilon: float = config[constants.epsilon]
        self.beta: float = config[constants.beta]
        self.tau: float = config[constants.tau]
        self.device = config[constants.device]
        self.actor: torch.nn.Module = config[constants.model_actor]
        self.critic: torch.nn.Module = config[constants.model_critic]
        self.target_actor: torch.nn.Module = pickle.loads(pickle.dumps(self.actor))  # clones the actor
        self.target_critic: torch.nn.Module = pickle.loads(pickle.dumps(self.critic))  # clones the critic
        self.optimiser_actor: torch.optim.optimizer.Optimizer = config[constants.optimiser_actor]
        self.optimiser_critic: torch.optim.optimizer.Optimizer = config[constants.optimiser_critic]
        self.ending_condition = config[constants.ending_condition]
        self.batch_size = 1024
        self.n_games = 1
        self.train_every = config[constants.train_every]
        self.train_n_times = config[constants.train_n_times]
        self.replay_buffer = PrioReplayBuffer(1000)

    # self.t_update_target_step = 0
    def required_properties(self):
        return [constants.input_dim,
                constants.output_dim,
                constants.max_t,
                constants.n_episodes,
                constants.gamma,
                constants.beta,
                constants.tau,
                constants.device,
                constants.model_actor,
                constants.model_critic,
                constants.optimiser_actor,
                constants.optimiser_critic,
                constants.ending_condition]

    def learn(self, experiences, indexes, is_values):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.stack(states)
        actions = torch.stack(actions).squeeze()
        rewards = torch.stack(rewards).unsqueeze(1)
        next_states = torch.stack(next_states)
        is_values = torch.from_numpy(is_values).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float, device=self.device).unsqueeze(1)
        suggested_target_next_actions = self.target_actor(next_states)
        suggested_actions = self.actor(states)
        target_critic_next_input = torch.cat((next_states, suggested_target_next_actions), dim=1)
        critic_input = torch.cat((states, actions), dim=1)
        suggested_critic_input = torch.cat((states, suggested_actions), dim=1)
        y = rewards + self.gamma * self.target_critic(target_critic_next_input) * (torch.ones_like(dones) - dones)  # sets 0 to the entries which are done
        states_value = self.critic(critic_input)
        suggested_actions_states_value = self.critic(suggested_critic_input)

        self.critic.train()
        self.actor.train()
        td_error = y.detach() - states_value
        self.replay_buffer.update_priorities(indexes, abs(td_error))
        loss_critic = 0.5 * (td_error).pow(2) * is_values.detach()
        self.optimiser_critic.zero_grad()
        loss_critic.mean().backward()
        torch.nn.utils.clip_grad_norm(self.critic.parameters(), 1)
        self.optimiser_critic.step()
        loss_actor = suggested_actions * suggested_actions_states_value
        self.optimiser_actor.zero_grad()
        loss_actor.mean().backward()
        self.optimiser_actor.step()
        self.critic.eval()
        self.actor.eval()
        self.soft_update(self.critic, self.target_critic, self.tau)
        self.soft_update(self.actor, self.target_actor, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

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
            next_state_list = []
            action_list = []
            reward_list = []
            done_list = []
            state = torch.tensor(env_info.vector_observations[0], dtype=torch.float, device=self.device)  # get the current state
            score = 0
            self.actor.eval()
            self.critic.eval()
            for t in range(self.max_t):
                action: torch.Tensor = self.actor(state.unsqueeze(dim=0))
                env_info = env.step(action.cpu().detach().numpy())[brain_name]  # send the action to the environment
                next_state = torch.tensor(env_info.vector_observations[0], dtype=torch.float, device=self.device)  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished
                state_list.append(state)
                action_list.append(action)
                reward_list.append(reward)
                done_list.append(1 if done else 0)
                next_state_list.append(next_state)
                state = next_state
                score += reward
                if done:
                    break
            # prepares the rewards
            rewards_array = self.calculate_discounted_rewards(reward_list)
            # todo implement GAE
            td_errors = self.calculate_td_errors(state_list, action_list, reward_list, next_state_list)
            # stores the episode en the replay buffer
            for i in range(len(action_list)):
                state = state_list[i]
                next_state = next_state_list[i]
                action = action_list[i]
                reward = torch.tensor(reward_list[i], device=self.device)
                done = done_list[i]
                self.replay_buffer.push((state, action, reward, next_state, done), abs(td_errors[i].item()))

            # train the agent
            if len(self.replay_buffer) > self.batch_size and i_episode % self.train_every == 0:
                experiences, indexes, is_values = self.replay_buffer.sample(self.batch_size)
                self.learn(experiences=experiences, indexes=indexes, is_values=is_values)
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

    def calculate_td_errors(self, state_list, action_list, reward_list, next_state_list):
        stacked_next_states = torch.stack(next_state_list, dim=0).to(self.device)
        stacked_states = torch.stack(state_list, dim=0).to(self.device)
        stacked_actions = torch.stack(action_list, dim=0).squeeze().to(self.device)
        concat_states = torch.cat([stacked_states, stacked_actions], dim=1).to(self.device)
        rewards = torch.tensor(reward_list).unsqueeze(dim=1).to(self.device)
        suggested_next_action = self.target_actor(stacked_next_states).to(self.device)
        concat_next_states = torch.cat([stacked_next_states, suggested_next_action], dim=1).to(self.device)
        td_errors = rewards + self.target_critic(concat_next_states) - self.critic(concat_states)
        return td_errors  # calculate the td-errors, maybe use GAE

    def calculate_discounted_rewards(self, reward_list: list) -> np.ndarray:
        rewards = reward_list.copy()
        rewards.reverse()
        previous_rewards = 0
        for i in range(len(rewards)):
            rewards[i] = rewards[i] + self.gamma * previous_rewards
            previous_rewards = rewards[i]
        rewards.reverse()
        rewards_array = np.asanyarray(rewards)
        return rewards_array

    def clipped_surrogate(self, old_log_probs, states, rewards_standardised, discount=0.995, epsilon=0.1, beta=0.01):
        # convert states to policy (or probability)

        actions, log_prob, entropy = self.model(states)
        # cost = torch.log(new_probs) * rewards_standardised
        ratio = torch.exp(log_prob - old_log_probs)
        cost = torch.min(ratio, torch.clamp(ratio, 1 - epsilon, 1 + epsilon))
        my_surrogate = -torch.mean(cost) * rewards_standardised  # + beta * entropy
        return my_surrogate
