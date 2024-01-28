import torch
from dataclasses import dataclass


@dataclass
class drnd_config:
    project: str = "sac_drnd"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path: str = "weights/sac_drnd.pt"

    state_dim: int = 17
    action_dim: int = 6

    actor_lr: float = 1e-3
    edac_init: bool = True
    critic_lr: float = 1e-3
    beta_lr: float = 1e-3
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 5e-3
    actor_lambda: float = 10.0
    critic_lambda: float = 10.0
    num_critics: int = 2
    critic_layernorm: bool = True

    drnd_learning_rate: float = 3e-4
    drnd_hidden_dim: int = 256
    drnd_embedding_dim: int = 32
    drnd_num_epochs: int = 100
    drnd_num_targets: int = 10
    drnd_alpha: float = 0.9

    dataset_name: str = "halfcheetah-medium-v2"  # "walker2d-medium-v2"
    batch_size: int = 1024
    # num_epochs: int = 150
    num_updates_on_epoch: int = 1000
    max_timesteps: int = 150000
    logging_interval: int = 10
    normalize_reward: bool = False
    
    group: str = dataset_name

    eval_episodes: int = 10
    eval_period: int = 50

    train_seed: int = 10
    eval_seed: int = 42
