from typing import Tuple, Dict
import torch
from torch import nn
import random

try:
    from rnd_utils import RunningMeanStd
    from modules import EnsembledLinear
except ModuleNotFoundError:
    from sac_drnd.rnd_utils import RunningMeanStd
    from sac_drnd.modules import EnsembledLinear


class PredictorNetwork(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int,
                 hidden_dim: int = 256,
                 num_hidden_layers: int = 4) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

        self.bilinear = nn.Bilinear(state_dim, action_dim, hidden_dim)
        
        layers = [nn.ReLU()]
        for _ in range(num_hidden_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, embedding_dim))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self,
                states: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        z = self.layers(self.bilinear(states, actions))
        return z


class EnsembledFiLM(nn.Module):
    '''
        Feature-wise Linear Modulation
    '''
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 ensemble_size: int) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.linear = EnsembledLinear(in_features, 2 * out_features, ensemble_size)
    
    def forward(self,
                states: torch.Tensor,
                h: torch.Tensor) -> torch.Tensor:
        gamma, beta = torch.split(self.linear(states), self.out_features, dim=-1)

        return gamma * h + beta


class EnsembledTargetNetwork(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int,
                 num_networks: int,
                 hidden_dim: int = 256,
                 num_hidden_layers: int = 4) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.num_networks = num_networks
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

        base_network = [
            EnsembledLinear(action_dim, hidden_dim, num_networks),
            nn.ReLU(),
        ]
        for _ in range(num_hidden_layers - 3):
            base_network.append(EnsembledLinear(hidden_dim, hidden_dim, num_networks))
            base_network.append(nn.ReLU())
        
        base_network.append(EnsembledLinear(hidden_dim, hidden_dim, num_networks))
        self.base_network = nn.Sequential(*base_network)

        self.film = EnsembledFiLM(state_dim, hidden_dim, num_networks)
        self.head = nn.Sequential(
            nn.ReLU(),
            EnsembledLinear(hidden_dim, embedding_dim, num_networks),
        )
    
    def forward(self,
                states: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        h = self.base_network(actions)
        h = self.film(states, h)
        z = self.head(h)
        return z


class DRND(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int,
                 num_target_networks: int,
                 alpha: float,
                 state_mean: torch.Tensor,
                 state_std: torch.Tensor,
                 action_mean: torch.Tensor,
                 action_std: torch.Tensor,
                 max_action: float = 1.0,
                 hidden_dim: int = 256,
                 num_hidden_layers: int = 4,
                 eps: float = 1e-6) -> None:
        super().__init__()

        self.state_mean, self.state_std = state_mean, state_std
        self.action_mean, self.action_std = action_mean, action_std
        self.eps = eps
        self.alpha = alpha

        self.loss_fn = nn.MSELoss(reduction="none")

        self.rms = RunningMeanStd()
        self.max_action = max_action
        
        self.predictor = PredictorNetwork(state_dim,
                                          action_dim,
                                          embedding_dim,
                                          hidden_dim,
                                          num_hidden_layers)
        self.predictor.train()

        self.target = EnsembledTargetNetwork(state_dim,
                                  action_dim,
                                  embedding_dim,
                                  num_target_networks,
                                  hidden_dim,
                                  num_hidden_layers)
        self.disable_target_grads()
        self.target.eval()
    
    def disable_target_grads(self):
        for p in self.target.parameters():
            p.requires_grad = False
    
    def normalize(self,
                  state: torch.Tensor,
                  action: torch.Tensor,
                  eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
        state = (state - self.state_mean) / (self.state_std + eps)
        action = (action - self.action_mean) / (self.action_std + eps)

        return state, action
    
    def forward(self,
                states: torch.Tensor,
                actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.target.eval()

        states, actions = self.normalize(states, actions)

        predictor_out = self.predictor(states, actions)
        target_out = self.target(states, actions)

        return predictor_out, target_out
    
    def loss(self,
             states: torch.Tensor,
             actions: torch.Tensor) -> torch.Tensor:
        '''
            outputs unreduced vector with shape as [batch_size, embedding_dim]
        '''
        predictor_out, target_out = self(states, actions)

        # sample the output of one of the target ensemble
        target_sample = random.choice(target_out)  # [batch_size, embedding_dim]

        loss = self.loss_fn(predictor_out, target_sample)
        return loss
    
    def drnd_bonus(self,
                  state: torch.Tensor,
                  action: torch.Tensor) -> torch.Tensor:
        predictor_out, target_out = self(state, action)

        target_mean: torch.Tensor = target_out.mean(dim=0)  # [batch_size, embedding_dim]
        B2: torch.Tensor = target_out.square().mean(dim=0)
        target_mean_squared = target_mean.square()

        b1 = self.loss_fn(predictor_out, target_mean).sum(dim=-1)
        b2 = ((predictor_out.square() - target_mean_squared) / (B2 - target_mean_squared)).sqrt().sum(dim=-1)

        return self.alpha * b1 + (1 - self.alpha) * b2
    
    def update_drnd(self,
                   states: torch.Tensor,
                   actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raw_loss = self.loss(states, actions).sum(dim=1)
        loss = raw_loss.mean(dim=0)

        self.rms.update(raw_loss)

        # made for logging
        random_actions = torch.rand_like(actions)
        random_actions = 2 * self.max_action * random_actions - self.max_action
        rnd_random = self.drnd_bonus(states, random_actions).mean()

        update_info = {
            "drnd/loss": loss.item(),
            "drnd/running_std": self.rms.std.item(),
            "drnd/data": loss / self.rms.std.item(),
            "drnd/random": rnd_random.item()
            }
        
        return loss, update_info


if __name__ == "__main__":
    # m = EnsembledFiLM(17, 32, 5)
    # states = torch.rand(4, 17)
    # h = torch.rand(4, 32)
    # print(m(states, h).shape)
    m = EnsembledTargetNetwork(17, 6, 32, 3)
    p = PredictorNetwork(17, 6, 32)
    state = torch.rand(4, 17)
    action = torch.rand(4, 6)

    target_out = m(state, action)
    target_sample = random.choice(target_out)
    print(target_sample.shape)
    # print(p(state, action).shape)
