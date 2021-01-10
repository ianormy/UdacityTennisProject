"""Multi-Agent DDPG
"""
from collections import deque
import numpy as np
from unity_ml_facade import UnityMlFacade
from policy import Policy
import torch
import os
from ou_noise import OUNoise
from replay_buffer import ReplayBuffer


class MultiAgentDDPG:
    def __init__(self,
                 env: [UnityMlFacade],
                 device,
                 seed,
                 verbose=1,
                 gamma=0.99,
                 actor_learning_rate=0.001,
                 critic_learning_rate=0.001,
                 buffer_size=100000,
                 batch_size=100,
                 snapshot_window=5,
                 hidden_layers_comma_sep='400,30'):
        self.env = env
        self.device = device
        self.seed = seed
        self.verbose = verbose
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.snapshot_window = snapshot_window
        self.policy_snapshots = deque(maxlen=self.snapshot_window)
        self.current_policy_snapshot = -1
        self.last_save = 0
        self.last_swap = 0
        self.action_size = self.env.action_space.shape[0] * self.env.num_agents
        self.state_size = self.env.observation_space.shape[0] * self.env.num_agents  # this should be 48
        hidden_layers = [int(layer_width) for layer_width in hidden_layers_comma_sep.split(',')]
        # create agent1
        self.player_policy = Policy(0, state_size=self.state_size, action_size=self.action_size,
                                    hidden_dims=hidden_layers, device=self.device,
                                    actor_learning_rate=actor_learning_rate,
                                    critic_learning_rate=critic_learning_rate,
                                    random_seed=seed)
        # create agent2
        self.opponent_policy = Policy(1, state_size=self.state_size, action_size=self.action_size,
                                      hidden_dims=hidden_layers, device=self.device,
                                      actor_learning_rate=actor_learning_rate,
                                      critic_learning_rate=critic_learning_rate,
                                      random_seed=seed)
        self.t_step = 0

    def learn_random(self, total_timesteps, callback=None):
        # start with random actions, just to test the loop
        action_size = self.env.action_space.shape[0]
        for i in range(1, 6):
            scores = np.zeros(self.env.num_agents)
            state, reward, done = self.env.reset()
            while True:
                actions = np.random.randn(self.env.num_agents, action_size)
                actions = np.clip(actions, -1, 1)
                next_states, rewards, dones, info = self.env.step(actions)
                scores += rewards
                states = next_states
                if np.any(dones):
                    break
            print('Score (max over agents) from episode {}: {} in steps: {}'.format(i, np.max(scores),
                                                                                    self.env.episode_step))

    def learn(self, total_timesteps, callback):
        ou_scale = 1.0  # initial scaling factor
        ou_decay = 0.9995  # decay of the scaling factor ou_scale
        ou_mu = 0.0  # asymptotic mean of the noise
        ou_theta = 0.15  # magnitude of the drift term
        ou_sigma = 0.20  # magnitude of the diffusion term
        # this slowly decreases to 0
        # create the noise process
        noise_process = OUNoise(self.action_size, ou_mu, ou_theta, ou_sigma)
        # create the replay buffer
        buffer = ReplayBuffer(seed=self.seed, action_size=self.action_size, buffer_size=self.buffer_size,
                              batch_size=self.batch_size, device=self.device)
        self.t_step = 0
        episode = 0
        while self.t_step < total_timesteps:
            callback.on_start_episode(episode)
            episode_scores = np.zeros(self.env.num_agents)
            states, _, _ = self.env.reset()
            scores = np.zeros(2)
            while True:
                states = np.reshape(states, (1, 48))  # reshape so we can feed both agents states to each agent
                # split into the states into the parts observed by each agent
                states_0 = states[0, :24].reshape((1, 24))
                states_1 = states[0, 24:].reshape((1, 24))
                # generate noise
                noise = ou_scale * noise_process.get_noise().reshape((1, 4))
                # split the noise into the parts for each agent
                noise_0 = noise[0, :2].reshape((1, 2))
                noise_1 = noise[0, 2:].reshape((1, 2))
                # determine actions for the unity agents from current sate, using noise for exploration
                actions_0 = self.player_policy.act(states_0, use_target=False, add_noise=True, noise_value=noise_0)\
                    .detach().cpu().numpy()
                actions_1 = self.opponent_policy.act(states_1, use_target=False, add_noise=True, noise_value=noise_1)\
                    .detach().cpu().numpy()
                actions = np.vstack((actions_0.flatten(), actions_1.flatten()))
                # take the action in the environment
                next_states, rewards, dones, info = self.env.step(actions)
                # store (S, A, R, S') info in the replay buffer (memory)
                buffer.add(states.flatten(), actions.flatten(), rewards, next_states.flatten(), dones)
                episode_scores += rewards
                states = next_states
                self.t_step += 1
                """
                Policy learning
                """
                ## train the agents if we have enough replays in the buffer
                if len(buffer) >= self.batch_size:
                    self.player_policy.learn(buffer.sample(), self.opponent_policy)
                    self.opponent_policy.learn(buffer.sample(), self.player_policy)
                if np.any(dones):
                    break
            if not callback.on_step(np.max(episode_scores), self.t_step):
                break
            # decrease the scaling factor of the noise
            ou_scale *= ou_decay
            episode += 1

    def save(self, model_folder):
        # Save trained  Actor and Critic network weights for agent 1
        an_filename = os.path.join(model_folder, "ddpg_player_actor.pth")
        torch.save(self.player_policy.actor.state_dict(), an_filename)
        cn_filename = "ddpg_player_critic.pth"
        torch.save(self.player_policy.critic.state_dict(), cn_filename)
        # Save trained  Actor and Critic network weights for agent 2
        an_filename = "ddpg_opponent_actor.pth"
        torch.save(self.opponent_policy.actor.state_dict(), an_filename)
        cn_filename = "ddpg_opponent_critic.pth"
        torch.save(self.opponent_policy.critic.state_dict(), cn_filename)

    def _save_snapshot(self, policy: Policy) -> None:
        """save a snapshot of the provided Policy weights"""
        weights = policy.get_weights()
        self.policy_snapshots.append(weights)
        self.current_policy_snapshot = weights

    def _swap_snapshots(self) -> None:
        if np.random.uniform() < (1 - self.play_against_current_self_ratio):
            x = np.random.randint(len(self.policy_snapshots))
            snapshot = self.policy_snapshots[x]
            self.current_opponent = x
        else:
            snapshot = self.current_policy_snapshot
            self.current_opponent = -1
        self.opponent_policy.load_weights(snapshot)

    def _step(self, states, actions, rewards, next_states, dones, info):
        """This method is called each training step with our (s,a,r,s',done)
          experience tuple.
          """
        #if self.t_step % self.save_steps == 0:
        #    self._save_snapshot(self.player_policy)
        #if self.t_step % self.swap_steps == 0:
        #    self._swap_snapshots()
