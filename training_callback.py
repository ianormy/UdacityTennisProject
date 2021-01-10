import time
from collections import deque
import numpy as np
import torch
import os


class TrainingCallback:
    def __init__(self, agent, reward_threshold=0.5, mean_num_items=100, print_every=200, save_every=25, verbose=1,
                 model_folder=None, results_folder=None):
        self.reward_threshold = reward_threshold
        self.mean_num_items = mean_num_items
        self.training_start = time.time()
        self.scores = []
        self.episode_seconds = []
        self.scores_window = deque(maxlen=mean_num_items)
        self.training_time = 0
        self.print_every = print_every
        self.save_every = save_every
        self.agent = agent
        self.verbose = verbose
        self.mean_reward = 0
        self.episode_start = None
        self.episode_num = -1
        self.model_folder = model_folder
        self.results_folder = results_folder

    def on_step(self, score, timesteps) -> bool:
        episode_elapsed = time.time() - self.episode_start
        self.episode_seconds.append(episode_elapsed)
        self.training_time += episode_elapsed
        self.scores.append(score)
        self.scores_window.append(score)
        self.mean_reward = np.mean(self.scores_window)
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.training_start))
        print('\rEpisode {:05} Elapsed: {} Average Score: {:.2f} Timesteps: {:07}'.format(
            self.episode_num, elapsed_str, self.mean_reward, timesteps), end="")
        if self.episode_num % self.print_every == 0:
            print('\rEpisode {:05} Elapsed: {} Average Score: {:.2f} Timesteps: {:07}'.format(
                self.episode_num, elapsed_str, self.mean_reward, timesteps))
        if self.episode_num % self.save_every == 0:
            self._save_training_models()
            self._save_results()
        continue_training = bool(self.mean_reward < self.reward_threshold)
        if not continue_training:
            self._save_final_models()
            self._save_results()
            if self.verbose > 0:
                print(
                    f"\nStopping training because the mean reward {self.mean_reward:.2f} "
                    f" is above the threshold {self.reward_threshold}"
                    f" at episode {self.episode_num}"
                )
        return continue_training

    def on_start_episode(self, episode_num):
        self.episode_start = time.time()
        self.episode_num = episode_num

    def _save_training_models(self):
        # save actors
        actor_state = {'epoch': self.episode_num, 'state_dict': self.agent.player_policy.actor.state_dict(),
                       'optimizer': self.agent.player_policy.actor_optimizer.state_dict(), }
        torch.save(actor_state, os.path.join(self.model_folder, 'checkpoint_actor_player.pth'))
        actor_opponent_state = {'epoch': self.episode_num,
                                'state_dict': self.agent.opponent_policy.actor.state_dict(),
                                'optimizer': self.agent.opponent_policy.actor_optimizer.state_dict(), }
        torch.save(actor_opponent_state, os.path.join(self.model_folder, 'checkpoint_actor_opponent.pth'))
        # save critics
        critic_state = {'epoch': self.episode_num, 'state_dict': self.agent.player_policy.critic.state_dict(),
                        'optimizer': self.agent.player_policy.critic_optimizer.state_dict(), }
        torch.save(critic_state, os.path.join(self.model_folder, 'checkpoint_critic_player.pth'))
        critic_opponent_state = {'epoch': self.episode_num,
                                 'state_dict': self.agent.opponent_policy.critic.state_dict(),
                                 'optimizer': self.agent.opponent_policy.critic_optimizer.state_dict(), }
        torch.save(critic_opponent_state, os.path.join(self.model_folder, 'checkpoint_critic_opponent.pth'))

    def _save_final_models(self):
        # save actors
        torch.save(self.agent.player_policy.actor.state_dict(),
                   os.path.join(self.model_folder, 'final_actor_player.pth'))
        torch.save(self.agent.opponent_policy.actor.state_dict(),
                   os.path.join(self.model_folder, 'final_actor_opponent.pth'))
        # save critics
        torch.save(self.agent.player_policy.critic.state_dict(),
                   os.path.join(self.model_folder, 'final_critic_player.pth'))
        torch.save(self.agent.opponent_policy.critic.state_dict(),
                   os.path.join(self.model_folder, 'final_critic_opponent.pth'))

    def _save_results(self):
        np.save(os.path.join(self.results_folder, 'maddpg_results'), self.scores)
        np.save(os.path.join(self.results_folder, 'maddpg_times'), self.episode_seconds)
