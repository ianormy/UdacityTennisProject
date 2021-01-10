"""Train a network

This uses an experiment framework that enables trying different
hyperparameters and storing the results.
"""
import argparse
import sys
import random
import os

from multi_agent_ddpg import MultiAgentDDPG
from training_callback import TrainingCallback
from unity_ml_facade import UnityMlFacade
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an agent using MADDPG and Unity ML agents",
        epilog="This uses an old version of Unity ML agents ")
    parser.add_argument(
        '--environment-port',
        type=int,
        default=5005,
        required=False,
        help='Unity environment port to use for communicating with Unity ML agent')
    parser.add_argument(
        '--total-timesteps',
        type=int,
        default=int(1e6),
        required=False,
        help='Limit the total number of timesteps for training')
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='maddpg-lr_0_0001-400_300-ts_250K',
        required=False,
        help='name to use for the experiment')
    parser.add_argument(
        '--experiments-root',
        type=str,
        default='.',
        required=False,
        help='path to the experiments root folder')
    parser.add_argument(
        '--executable-path',
        type=str,
        default='D:/Source/Udacity/TennisMultiAgent/Tennis_Windows_x86_64/Tennis.exe',
        required=False,
        help='path to the Tennis executable')
    parser.add_argument(
        '--seed',
        type=int,
        default=-1,
        required=False,
        help='seed to use for training. Default of -1 means use a random seed')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        required=False,
        help='Size of batches to use for training')
    parser.add_argument(
        '--snapshot-window',
        type=int,
        default=5,
        required=False,
        help='the size of the sliding window of past snapshots from which the opponents of the agent are sampled')
    parser.add_argument(
        '--hidden-layers',
        type=str,
        default='400,300',
        required=False,
        help='shape of hidden layers')
    parser.add_argument(
        '--actor-learning-rate',
        type=float,
        default=1e-4,
        required=False,
        help='actor learning rate')
    parser.add_argument(
        '--critic-learning-rate',
        type=float,
        default=1e-3,
        required=False,
        help='critic learning rate')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        required=False,
        help='Greedy strategy gamma value')
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=1000000,
        required=False,
        help='Size of the buffer for replay memory')
    return parser, parser.parse_args()


def main():
    # parse arguments
    parser, args = parse_args()
    # exit and show help if no arguments provided at all
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    # create all the folders needed for the experiment
    experiment_folder = os.path.join(args.experiments_root, args.experiment_name)
    os.makedirs(experiment_folder, exist_ok=True)
    model_folder = os.path.join(experiment_folder, 'model')
    os.makedirs(model_folder, exist_ok=True)
    results_folder = os.path.join(experiment_folder, 'results')
    os.makedirs(results_folder, exist_ok=True)
    # create the environment
    env = UnityMlFacade(executable_path=args.executable_path, seed=args.seed,
                        environment_port=args.environment_port)
    seed = random.randint(0, int(1e6)) if args.seed == -1 else args.seed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiAgentDDPG(
        env,
        seed=seed,
        verbose=1,
        device=device,
        gamma=args.gamma,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        buffer_size=args.buffer_size,
        hidden_layers_comma_sep=args.hidden_layers,
        batch_size=args.batch_size)
    # setup the callback
    callback = TrainingCallback(model, reward_threshold=0.5, model_folder=model_folder, results_folder=results_folder)
    # save all the hyperparameters for this experiment
    with open(os.path.join(experiment_folder, 'hyperparameters.txt'), 'w') as f:
        f.write('actor learning rate: {}\n'.format(args.actor_learning_rate))
        f.write('critic learning rate: {}\n'.format(args.critic_learning_rate))
        f.write('hidden layers: [{}]\n'.format(args.policy_layers))
        f.write('seed: {}\n'.format(seed))
        f.write('batch size: {}\n'.format(args.batch_size))
        f.write('buffer size: {}\n'.format(args.buffer_size))
        f.write('gamma: {}\n'.format(args.gamma))
        f.write('total timesteps: {}\n'.format(args.total_timesteps))
    # run the experiment
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback
    )
    # save the final model
    model.save(os.path.join(model_folder))


if __name__ == '__main__':
    main()
