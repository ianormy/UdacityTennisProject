"""Policy

Used for storing the actor and critic networks
together with their optimisers.
"""
from model import Actor, Critic
from torch.optim import Adam
import torch
import torch.nn.functional as F
import numpy as np
from replay_buffer import ReplayBuffer
from utilities import ensure_is_tensor


class Policy:
    def __init__(self,
                 agent_index,
                 state_size,
                 action_size,
                 hidden_dims,
                 device,
                 random_seed=7,
                 buffer_size=1000000,
                 batch_size=100,
                 actor_learning_rate=1e-3,
                 gamma=0.99,
                 tau=1e-3,
                 critic_learning_rate=1e-4):
        super(Policy, self).__init__()
        self.agent_index = agent_index
        self.tau = tau
        self.gamma = gamma
        self.seed = random_seed
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.action_size = action_size
        self.single_agent_state_size = state_size // 2
        self.single_agent_action_size = action_size // 2
        # actor networks - work as single agents
        self.actor = Actor(state_size=self.single_agent_state_size, action_size=self.single_agent_action_size,
                           seed=self.seed, hidden_dims=hidden_dims).to(device)
        self.target_actor = Actor(state_size=self.single_agent_state_size, action_size=self.single_agent_action_size,
                                  seed=self.seed, hidden_dims=hidden_dims).to(device)
        # set actor and target_actor with same weights & biases
        for local_param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(local_param.data)
        # critic networks - combine both agents
        self.critic = Critic(state_size=state_size, action_size=action_size, seed=self.seed,
                             hidden_dims=hidden_dims).to(device)
        self.target_critic = Critic(state_size=state_size, action_size=action_size, seed=self.seed,
                                    hidden_dims=hidden_dims).to(device)
        # set critic_local and critic_target with same weights & biases
        for local_param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(local_param.data)
        # optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_learning_rate, weight_decay=0)

        # Replay memory
        self.memory = ReplayBuffer(action_size=action_size, buffer_size=self.buffer_size, batch_size=self.batch_size,
                                   seed=self.seed, device=self.device)
        self.t_update = 0

    def get_weights(self):
        """get the weights for the actor and critic models"""
        return self.actor.state_dict(), self.target_actor.state_dict(), \
               self.critic.state_dict(), self.target_critic.state_dict()

    def load_weights(self, values):
        """load the weights for the actor and critic models"""
        w1, w2, w3, w4 = values
        self.actor.load_state_dict(w1)
        self.target_actor.load_state_dict(w2)
        self.critic.load_state_dict(w3)
        self.target_critic.load_state_dict(w4)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        if self.num_agents > 1:
            for agent in range(self.num_agents):
                self.memory.add(states[agent, :], actions[agent, :], rewards[agent], next_states[agent, :],
                                dones[agent])
        else:
            self.memory.add(states, actions, rewards, next_states, dones)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def act(self, state, use_target=False, add_noise=True, noise_value=None):
        """Returns actions for given state as per current policy.

        Arguments:
            state (Tensor): input state
            use_target (bool): if True then use the target actor network, otherwise
                use the local one
            add_noise (bool): if True then add noise to the actions obtained
            noise_value (float): noise value to add (if adding noise)
        Returns:
            action (Tensor): action of shape (action_size)  # (2)
        """
        state = ensure_is_tensor(state, self.device)
        if use_target:
            actor_net = self.target_actor
        else:
            actor_net = self.actor
        actor_net.eval()
        with torch.no_grad():
            action = actor_net(state)
            if add_noise:
                action = action.cpu().data.numpy()
                action = np.clip(action + noise_value, -1, 1)
                action = ensure_is_tensor(action, self.device)
        actor_net.train()
        return action

    def learn(self, experiences, other_agent):
        """Update policy and value parameters using given batch of experience tuples.
         Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
         where:
             actor_target(state) -> action
             critic_target(state, action) -> Q-value

         Arguments:
             experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
             other_agent (Policy): the other agent
         """
        states, actions, rewards, next_states, dones = experiences
        self.t_update += 1
        if self.agent_index == 0:
            # states, actions, next_actions of the agent
            states_self = ensure_is_tensor(states[:, :self.single_agent_state_size], self.device)
            action_self = ensure_is_tensor(actions[:, :self.single_agent_action_size], self.device)
            next_states_self = ensure_is_tensor(next_states[:, :self.single_agent_state_size], self.device)
            # states, actions, next_actions of the other agent
            states_other = ensure_is_tensor(states[:, self.single_agent_state_size:], self.device)
            action_other = ensure_is_tensor(actions[:, self.single_agent_action_size:], self.device)
            next_states_other = ensure_is_tensor(next_states[:, self.single_agent_state_size:], self.device)
            # rewards and dones
            rewards = ensure_is_tensor(rewards[:, 0].reshape((-1, 1)), self.device)
            dones = ensure_is_tensor(dones[:, 0].reshape((-1, 1)), self.device)
        elif self.agent_index == 1:
            # states, actions, next_actions of the agent
            states_self = ensure_is_tensor(states[:, self.single_agent_state_size:], self.device)
            action_self = ensure_is_tensor(actions[:, self.single_agent_action_size:], self.device)
            next_states_self = ensure_is_tensor(next_states[:, self.single_agent_state_size:], self.device)
            # states, actions, next_actions of the other agent
            states_other = ensure_is_tensor(states[:, :self.single_agent_state_size], self.device)
            action_other = ensure_is_tensor(actions[:, :self.single_agent_action_size], self.device)
            next_states_other = ensure_is_tensor(next_states[:, :self.single_agent_state_size], self.device)
            # rewards and dones
            rewards = ensure_is_tensor(rewards[:, 1].reshape((-1, 1)), self.device)
            dones = ensure_is_tensor(dones[:, 1].reshape((-1, 1)), self.device)
        # s, a, s' for both agents
        states = ensure_is_tensor(states, self.device)
        actions = ensure_is_tensor(actions, self.device)
        next_states = ensure_is_tensor(next_states, self.device)
        # ---------------------------- update critic ---------------------------- #
        next_actions_self = self.act(next_states_self, use_target=True, add_noise=False)
        next_actions_other = other_agent.act(next_states_other, use_target=True, add_noise=False)
        # combine the next actions from both agents
        if self.agent_index == 0:
            actions_next = torch.cat([next_actions_self, next_actions_other], dim=1).float().detach().to(
                self.device)
        elif self.agent_index == 1:
            actions_next = torch.cat([next_actions_other, next_actions_self], dim=1).float().detach().to(
                self.device)
        # Get predicted next-state actions and Q values from target models
        self.target_critic.eval()
        with torch.no_grad():
            Q_targets_next = self.target_critic(next_states, actions_next).detach().to(self.device)
        self.target_critic.train()
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        # get the current action-value for the states and actions
        Q_expected = self.critic(states, actions)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        # Compute critic loss
        critic_loss = F.smooth_l1_loss(Q_expected, Q_targets.detach())
        # back propagate through the network
        critic_loss.backward()
        self.critic_optimizer.step()
        # ---------------------------- update actor ---------------------------- #
        if self.agent_index == 0:
            actions_pred = torch.cat([self.actor(states_self),
                                      other_agent.act(states_other, use_target=False, add_noise=False)],
                                     dim=1)
        elif self.agent_index == 1:
            actions_pred = torch.cat([other_agent.act(states_other, use_target=False, add_noise=False),
                                      self.actor(states_self)],
                                     dim=1)
        # Compute actor loss and minimize it
        self.actor_optimizer.zero_grad()
        actor_loss = - self.critic(states, actions_pred).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic, self.target_critic, self.tau)
        self.soft_update(self.actor, self.target_actor, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Arguments:
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
