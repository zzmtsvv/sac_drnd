from tqdm import trange, tqdm
from typing import Tuple
import numpy as np
import torch
from torch.optim import Adam
# import gym

from dataset import ReplayBuffer
from drnd_modules import DRND
from modules import Actor, EnsembledCritic
from sac_drnd import SAC_DRND
from config import drnd_config
from utils import seed_everything, make_dir

import wandb

# import d4rl


class SACDRNDTrainer:
    def __init__(self,
                 cfg=drnd_config) -> None:
        make_dir("weights")

        self.device = cfg.device
        self.batch_size = cfg.batch_size
        self.cfg = cfg

        # self.eval_env = gym.make(cfg.dataset_name)
        # self.eval_env.seed(cfg.eval_seed)
        # d4rl_dataset = d4rl.qlearning_dataset(self.eval_env)
        
        # self.state_dim = self.eval_env.observation_space.shape[0]
        # self.action_dim = self.eval_env.action_space.shape[0]
        self.state_dim = cfg.state_dim
        self.action_dim = cfg.action_dim

        self.buffer = ReplayBuffer(self.state_dim, self.action_dim)
        # self.buffer.from_d4rl(d4rl_dataset)
        self.buffer.from_json(cfg.dataset_name)

        seed_everything(cfg.train_seed)
    
    def train_drnd(self) -> DRND:
        (self.state_mean, self.state_std), (self.action_mean, self.action_std) = self.buffer.get_moments()

        drnd = DRND(self.state_dim,
                  self.action_dim,
                  self.cfg.drnd_embedding_dim,
                  self.cfg.drnd_num_targets,
                  self.cfg.drnd_alpha,
                  self.state_mean,
                  self.state_std,
                  self.action_mean,
                  self.action_std,
                  hidden_dim=self.cfg.drnd_hidden_dim).to(self.device)
        drnd_optim = Adam(drnd.predictor.parameters(), lr=self.cfg.drnd_learning_rate)

        for epoch in trange(self.cfg.drnd_num_epochs, desc="DRND Epochs"):

            for _ in trange(self.cfg.num_updates_on_epoch, desc="DRND Iterations"):
                states, actions, _, _, _, = self.buffer.sample(self.batch_size)
                states, actions = [x.to(self.device) for x in (states, actions)]

                loss, update_info = drnd.update_drnd(states, actions)
                drnd_optim.zero_grad()
                loss.backward()
                drnd_optim.step()

                wandb.log(update_info)
        
        return drnd
    
    def train(self):
        '''
            - setup drnd and wandb
            - train drnd
            - setup sac drnd
            - train sac drnd

        '''
        run_name = f"sac_drnd_" + str(self.cfg.train_seed)
        print(f"Training starts on {self.cfg.device} 🚀")


        with wandb.init(project=self.cfg.project, group=self.cfg.group, name=run_name, job_type="offline_training"):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})

            drnd = self.train_drnd()
            drnd.eval()

            actor = Actor(self.state_dim, self.action_dim, self.cfg.hidden_dim)
            actor_optim = Adam(actor.parameters(), lr=self.cfg.actor_lr)
            critic = EnsembledCritic(self.state_dim, self.action_dim, self.cfg.hidden_dim, layer_norm=self.cfg.critic_layernorm)
            critic_optim = Adam(critic.parameters(), lr=self.cfg.critic_lr)

            self.sac_drnd = SAC_DRND(actor,
                                   actor_optim,
                                   critic,
                                   critic_optim,
                                   drnd,
                                   self.cfg.actor_lambda,
                                   self.cfg.critic_lambda,
                                   self.cfg.beta_lr,
                                   self.cfg.gamma,
                                   self.cfg.tau,
                                   self.device)
            
            for t in tqdm(range(self.cfg.max_timesteps), desc="SAC DRND steps"):
                batch = self.buffer.sample(self.batch_size)

                states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]

                logging_dict = self.sac_drnd.train_offline_step(states,
                                                               actions,
                                                               rewards,
                                                               next_states,
                                                               dones)
                
                wandb.log(logging_dict, step=self.sac_drnd.total_iterations)
            
            # for epoch in trange(self.cfg.num_epochs, desc="Offline SAC Epochs"):
            #     update_info_total = {
            #         "sac_offline/actor_loss": 0,
            #         "sac_offline/actor_batch_entropy": 0,
            #         "sac_offline/drnd_policy": 0,
            #         "sac_offline/drnd_random": 0,
            #         "sac_offline/critic_loss": 0,
            #         "sac_offline/q_mean": 0
            #         }

            #     for _ in range(self.cfg.num_updates_on_epoch):
            #         state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

            #         update_info = self.sac_drnd.train_offline_step(state,
            #                                                       action,
            #                                                       reward,
            #                                                       next_state,
            #                                                       done)
                    
            #         for k, v in update_info.items():
            #             update_info_total[k] += v
                
            #     for k, v in update_info_total.items():
            #         update_info_total[k] /= self.cfg.num_updates_on_epoch
                
            #     wandb.log(update_info)

            #     if epoch % self.cfg.eval_period == 0 or epoch == self.cfg.num_epochs - 1:
                    
            #         eval_returns = self.eval_actor()
                    
            #         normalized_score = self.eval_env.get_normalized_score(eval_returns) * 100.0

            #         wandb.log({
            #             "eval/return_mean": np.mean(eval_returns),
            #             "eval/return_std": np.std(eval_returns),
            #             "eval/normalized_score_mean": np.mean(normalized_score),
            #             "eval/normalized_score_std": np.std(normalized_score)
            #         })
        
        wandb.finish()

    # @torch.no_grad()
    # def eval_actor(self) -> np.ndarray:
    #     self.eval_env.seed(self.cfg.eval_seed)
    #     self.sac_drnd.actor.eval()
    #     episode_rewards = []
        
    #     for _ in range(self.cfg.eval_episodes):

    #         state, done = self.eval_env.reset(), False
    #         episode_reward = 0.0

    #         while not done:
    #             action = self.sac_drnd.actor.act(state, self.device)
    #             state, reward, done, _ = self.eval_env.step(action)
    #             episode_reward += reward
    #         episode_rewards.append(episode_reward)

    #     self.sac_drnd.actor.train()
    #     return np.array(episode_rewards)
    
    def save(self):
        state_dict = self.sac_drnd.state_dict()
        torch.save(state_dict, self.cfg.checkpoint_path)

    # def load(self, map_location: str = "cpu"):
    #     state_dict = torch.load(self.cfg.checkpoint_path, map_location=map_location)

    #     actor = Actor(self.state_dim, self.action_dim, self.cfg.hidden_dim)
    #     actor_optim = Adam(actor.parameters(), lr=self.cfg.actor_lr)
    #     critic = EnsembledCritic(self.state_dim, self.action_dim, self.cfg.hidden_dim, layer_norm=self.cfg.critic_layernorm)
    #     critic_optim = Adam(critic.parameters(), lr=self.cfg.critic_lr)

    #     rnd = DRND(self.state_dim,
    #               self.action_dim,
    #               self.cfg.rnd_embedding_dim,
    #               self.state_mean,
    #               self.state_std,
    #               self.action_mean,
    #               self.action_std,
    #               hidden_dim=self.cfg.rnd_hidden_dim)
        
    #     self.sac_drnd = SAC_DRND(actor,
    #                            actor_optim,
    #                            critic,
    #                            critic_optim,
    #                            rnd,
    #                            self.cfg.actor_alpha,
    #                            self.cfg.critic_alpha,
    #                            self.cfg.beta_lr,
    #                            self.cfg.gamma,
    #                            self.cfg.tau,
    #                            self.device)
        
    #     self.sac_drnd.load_state_dict(state_dict)
