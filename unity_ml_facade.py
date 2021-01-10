"""This uses the facade pattern to interface with Unity ML
"""
from typing import Any, Tuple, Dict, Optional
import random
import gym
from gym.spaces import Box
import numpy as np
from unityagents import UnityEnvironment, BrainInfo, BrainParameters


class UnityMlFacade(gym.Env):
    def __init__(
            self,
            *args,
            executable_path,
            train_mode=True,
            seed=None,
            environment_port=None,
            **kwargs):
        """A facade between Unity ML and OpenAI gym.

        Args:
            *args: arguments which will be directly passed through to the Unity environment.
            executable_path (str): path to the Unity executable
            train_mode (bool): If True then set the unity environment to train mode, otherwise leave it in eval mode
            seed (int): the value of the seed to use for the environment. If none then a random seed will be
                generated and used
            environment_port: port of the environment. This enables multiple environments to be run at the same time
            **kwargs: keyword arguments that will be passed on to the Unity environment.
        """
        self.train_mode = train_mode
        self.unity_env, self.brain_name, self.brain = self._setup_unity(
            *args,
            path=executable_path,
            environment_port=environment_port,
            seed=seed,
            no_graphics=True,
            **kwargs)
        self.action_space, self.observation_space, self.reward_range = self._get_environment_specs(self.brain)
        # reset the environment to discover how many agents there are
        info = self.unity_env.reset(train_mode=self.train_mode)[self.brain_name]
        self.num_agents = len(info.agents)
        self.state_size = len(self.observation_space.high)
        # initialise the episode
        self.episode_step = 0
        self.episode_reward = np.zeros(self.num_agents)

    def step(self, action) -> Tuple[Any, float, bool, Dict[str, Any]]:
        brain_info = self.unity_env.step(action)[self.brain_name]
        state, reward, done = self._parse_brain_info(brain_info)
        self.episode_reward += reward
        info = (
            dict(episode=dict(
                r=self.episode_reward,
                l=self.episode_step))
            if done else dict())
        self.episode_step += 1
        return state, reward, done, info

    def reset(self) -> np.ndarray:
        brain_info = self.unity_env.reset(train_mode=self.train_mode)[self.brain_name]
        self.episode_step = 0
        self.episode_reward = np.zeros(self.num_agents)
        return self._parse_brain_info(brain_info)

    def render(self, mode='human') -> None:
        """not implemented"""
        pass

    def close(self):
        self.unity_env.close()

    def _parse_brain_info(self, info: BrainInfo):
        """Extract the next state, reward and done information from an environment brain.

        Arguments:
            info (BrainInfo): Unity environment brain

        Returns:
            next_state, reward, done (tuple):
        """
        next_state = info.vector_observations
        reward = info.rewards
        done = info.local_done
        return next_state, reward, done

    @staticmethod
    def _setup_unity(*args, path, environment_port=None, seed=None, **kwargs):
        """Setup a Unity environment and return it and its brain.

        Arguments:
            *args: parameters that will be passed through to the UnityEnvironment
            path (str): path to the Unity environment executable
            environment_port (int): port to use for communication with the Unity environment. If none is provided
                then the default value of 5005 will be used.
            seed (int): random seed. If none is provided then a random value will be generated
                in the range 0-1,000,000
            **kwargs: keyword arguments that will be passed through to the UnityEnvironment

        Returns:
            unity_env (UnityEnvironment): Unity environment
            brain_name (str): name of the brain
            brain (BrainParameters): brain parameters
        """
        kwargs['file_name'] = path
        kwargs['seed'] = random.randint(0, int(1e6)) if not seed else seed
        if environment_port:
            kwargs['base_port'] = environment_port
        unity_env = UnityEnvironment(*args, **kwargs)
        brain_name = unity_env.brain_names[0]
        brain = unity_env.brains[brain_name]
        return unity_env, brain_name, brain

    @staticmethod
    def _get_environment_specs(brain: BrainParameters):
        """Extract the action space, observation space and reward range info from an environment brain.

        Arguments:
            brain (BrainParameters): brain parameters

        Returns:
            action_space (Box): action space
            observation_space (Box): observation (state) space
            reward_range (Tuple(float, float)): tuple of the minimum and maximum values
        """
        action_space_size = brain.vector_action_space_size
        observation_space_size = brain.vector_observation_space_size
        observation_space_size *= brain.num_stacked_vector_observations
        # all actions are in the range -1.0 to 1.0
        action_space = Box(
            low=np.array(action_space_size * [-1.0]),
            high=np.array(action_space_size * [1.0]))
        observation_space = Box(
            low=np.array(observation_space_size * [-float('inf')]),
            high=np.array(observation_space_size * [float('inf')]))
        reward_range = (-float('inf'), float('inf'))

        return action_space, observation_space, reward_range
