import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    f = layer.weight.data.size()[0]
    layer.weight.data.uniform_(-1.0 / np.sqrt(f), 1.0 / np.sqrt(f))
    layer.bias.data.fill_(0.1)
    return layer


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_dims=(400, 300)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_layers = nn.ModuleList([])
        self.hidden_layers.append(nn.Linear(state_size, hidden_dims[0]))
        for h1, h2 in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.hidden_layers.append(nn.Linear(h1, h2))
        self.output_layer = nn.Linear(hidden_dims[-1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.hidden_layers:
            # fan_in
            f = layer.weight.data.size()[0]
            layer.weight.data.uniform_(-1.0 / np.sqrt(f), 1.0 / np.sqrt(f))
            layer.bias.data.fill_(0.1)
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)
        self.output_layer.bias.data.fill_(0.1)

    def forward(self, x):
        """Build an actor (policy) network that maps states -> actions."""
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return torch.tanh(self.output_layer(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_dims=(400, 300)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0]+action_size, hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        # initialise the weights
        f1 = self.fcs1.weight.data.size()[0]
        f2 = self.fc2.weight.data.size()[0]
        self.fcs1.weight.data.uniform_(-1.0 / np.sqrt(f1), 1.0 / np.sqrt(f1))
        self.fc2.weight.data.uniform_(-1.0 / np.sqrt(f2), 1.0 / np.sqrt(f2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        # initialise the biases
        self.fcs1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
        self.fc3.bias.data.fill_(0.1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
