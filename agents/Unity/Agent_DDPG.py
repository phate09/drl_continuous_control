import pickle
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim.optimizer
from tensorboardX import SummaryWriter

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
        self.max_t: int = config[constants.max_t]
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
        self.batch_size = config[constants.batch_size]
        self.n_games = 1
        self.beta_start = 0.4
        self.beta_end = 1
        self.train_every = config[constants.train_every]
        self.train_n_times = config[constants.train_n_times]
        self.replay_buffer = PrioReplayBuffer(config[constants.buffer_size], alpha=0.6)

    # self.t_update_target_step = 0
    def required_properties(self):
        return [constants.input_dim,
                constants.output_dim,
                constants.max_t,
                constants.n_episodes,
                constants.gamma,
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
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        is_values = torch.from_numpy(is_values).to(self.device)
        dones = torch.stack(dones).float()
        target_mu_sprime = self.target_actor(next_states)
        mu_s = self.actor(states)
        target_sprime_target_mu_sprime = torch.cat((next_states, target_mu_sprime), dim=1)
        s_a = torch.cat((states, actions), dim=1)
        s_mu_s = torch.cat((states, mu_s), dim=1)
        y = rewards + self.gamma * self.target_critic(target_sprime_target_mu_sprime) * (torch.ones_like(dones) - dones)  # sets 0 to the entries which are done
        Qs_a = self.critic(s_a)
        Qs_mu_s = self.critic(s_mu_s)

        self.critic.train()
        self.actor.train()
        td_error = y.detach() - Qs_a
        self.replay_buffer.update_priorities(indexes, abs(td_error))
        loss_critic = td_error.pow(2) * is_values.detach()
        self.optimiser_critic.zero_grad()
        loss_critic.mean().backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.optimiser_critic.step()
        loss_actor = -Qs_mu_s  # gradient ascent # mu_s *
        self.optimiser_actor.zero_grad()
        loss_actor.mean().backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
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

    def train(self, env, writer: SummaryWriter, ending_condition):
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
        i_steps = 0
        for i_episode in range(self.n_episodes):
            env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
            # Reset the memory of the agent
            state_list = []
            next_state_list = []
            action_list = []
            reward_list = []
            done_list = []
            states = torch.tensor(env_info.vector_observations, dtype=torch.float, device=self.device)  # get the current state
            score = 0
            self.actor.eval()
            self.critic.eval()
            for t in range(self.max_t):
                actions: torch.Tensor = self.actor(states)
                noise_upper = 1
                noise_lower = -1
                with torch.no_grad():
                    noise = torch.normal(torch.zeros_like(actions), torch.ones_like(actions) * 0.2)
                # noise = ((noise_upper - noise_lower) * torch.rand_like(actions) + noise_lower)  # adds exploratory noise
                actions = torch.clamp(actions + noise, -1, 1)  # clips the action to the allowed boundaries
                env_info = env.step(actions.cpu().detach().numpy())[brain_name]  # send the action to the environment
                next_states = torch.tensor(env_info.vector_observations, dtype=torch.float, device=self.device)  # get the next state
                rewards = torch.tensor(env_info.rewards, device=self.device).unsqueeze(dim=1)  # get the reward
                dones = torch.tensor(env_info.local_done, dtype=torch.uint8, device=self.device).unsqueeze(dim=1)  # see if episode has finished
                state_list.append(states)
                action_list.append(actions)
                reward_list.append(rewards)
                done_list.append(dones)
                next_state_list.append(next_states)
                td_error = self.calculate_td_errors(states, actions, rewards, next_states)
                for i in range(states.size()[0]):
                    self.replay_buffer.push((states[i], actions[i], rewards[i], next_states[i], dones[i]), abs(td_error[i].item()))
                # train the agent
                if len(self.replay_buffer) > self.batch_size and i_steps != 0 and i_steps % self.train_every == 0:
                    beta = (self.beta_end - self.beta_start) * i_episode / self.n_episodes + self.beta_start
                    for i in range(self.train_n_times):
                        experiences, indexes, is_values = self.replay_buffer.sample(self.batch_size, beta=beta)
                        self.learn(experiences=experiences, indexes=indexes, is_values=is_values)
                states = next_states
                score += rewards.mean().item()
                i_steps += 1
                if dones.any():
                    break
            # prepares the rewards
            # rewards_array = self.calculate_discounted_rewards(reward_list)
            # todo implement GAE
            # td_errors = self.calculate_td_errors(state_list, action_list, reward_list, next_state_list)
            # # stores the episode en the replay buffer
            # for i in range(len(action_list)):
            #     state = state_list[i]
            #     next_state = next_state_list[i]
            #     action = action_list[i]
            #     reward = torch.tensor(reward_list[i], device=self.device)
            #     done = done_list[i]
            #     self.replay_buffer.push((state, action, reward, next_state, done), abs(td_errors[i].item()))

            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            writer.add_scalar('data/score', score, i_episode)
            writer.add_scalar('data/score_average', np.mean(scores_window), i_episode)
            writer.flush()
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

    def calculate_td_errors(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        self.target_actor.eval()
        self.target_critic.eval()
        concat_states = torch.cat([states, actions], dim=1).to(self.device)
        suggested_next_action = self.target_actor(next_states).to(self.device)
        concat_next_states = torch.cat([next_states, suggested_next_action], dim=1).to(self.device)
        td_errors = rewards + self.gamma * self.target_critic(concat_next_states) - self.critic(concat_states)
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
