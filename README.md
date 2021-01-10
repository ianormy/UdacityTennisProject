# Udacity Tennis Project
Udacity Tennis Project

## Introduction
This is a solution for the third project of the [Udacity deep reinforcement learning course](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). It includes a script to train an agent using the MADDPG algorithm to play tennis.

![Trained Agent](https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif)

## Problem description
The task is episodic. In this environment, two agents control rackets to bounce a ball over a net. The goal of each agent is to keep the ball in play.

- Rewards:
  - If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01
  - After each episode, add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. Take the maximum of these 2 scores.
  - This yields a single **score** for each episode.
- Input state:
  - 24 continuous variables for each agent which consist of 3 frames of 8 variables each corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.
  - For 2 agents this means 48 continuous variables.
- Actions:
  - 2 continuous variables for each agent corresponding to movement toward (or away from) the net, and jumping with values in the range [-1.0, 1.0]
  - For 2 agents this means 4 continuous variables
- Goal:
  - Get an average **score** of at least +0.5 over 100 consecutive episodes
- Environment:
  -  The environment that is used is a dual agent provided by Udacity. 

## Solution
The problem is solved with MADDPG.
 
## Setup project
Unfortunately, the Unity ML environment used by Udacity for this project is a very early version that is a few years old - [v0.4](https://github.com/Unity-Technologies/ml-agents/releases/tag/0.4.0). This makes it extremely difficult to set things up, particularly in a Windows environment. Please see the separate guide I have provided on how to do this: [setup project](Setup.md).

## Training an agent
I have implemented an experiment based framework that allows for exploration of different hyperparameters when training a model. The parameters that you can specify are these:

- **actor-learning-rate** the learning rate to use for training the actor part of the model. Default is 0.0001.
- **critic-learning-rate** the learning rate to use for training the critic part of the model. Default is 0.001.
- **batch-size** the size of batches that are sampled and used to train the model. Default is 128.
- **buffer-size** the size of the replay buffer. Default is 100,000.
- **total-timesteps** the total number of timesteps to train for. Default is 100,000.
- **seed** random seed to use. Default is -1 which means generate a random value.
- **environment-port** this is the port number used for communication with the Unity environment. If you want to have more than one agent running at the same time you would specify a different port for each of them. Default is 5005.
- **hidden-layers** the hidden layers to use in the neural networks. Specified as a comma separated list. Default is "400,300".
- **executable-path** the path to the executable to run. Please see the notes about [setup project](Setup.md) to help with this.
- **experiments-root** the root folder that experiment output will be written to.
- **experiment-name** the name of the experiment. A subfolder will be created to the 'experiments-root' folder with this name and all output from this experiment will be written to it.
- **reward-threshold** the reward value that the solution must attain over 100 consecutive episodes. Default is 0.5.
- **gamma** the gamma value used for greedy strategy. Default is 0.99.

Please note that the default values chosen for these parameters correspond to the values used in the original [DDPG paper](https://arxiv.org/pdf/1509.02971.pdf) (see the **Experiment details** section of the paper).

### Example command line
Here is an example command line:

```python train_agent.py --experiments-root "D:\Data\Udacity\experiments\Tennis" --experiment-name maddpg-lr_0_0001-400_300-ts_250K --actor-learning-rate 0.0001 --critic-learning-rate 0.001 --total-timesteps 250000 --environment-port 5005 --policy-layers "400,300"```
