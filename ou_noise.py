import numpy as np
import random
import copy


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:

    def __init__(self, size, seed, mu=0.0, theta=0.1, sigma=.5, sigma_min=0.05, sigma_decay=.99):
        """Initialise parameters and the noise process"""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.seed = random.seed(seed)
        self.size = size
        self.reset()
        self.state = mu * np.ones(size)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        # Reduce  sigma from initial value to min
        self.sigma = max(self.sigma_min, self.sigma*self.sigma_decay)

    def get_noise(self):
        """
        Generate a noise vector of dimension = self.size
        """
        self.state += self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        return self.state

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
