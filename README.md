<!-- https://wandb.ai/zzmtsvv/sac_drnd/runs/d03hrwpr?workspace=user-zzmtsvv -->

# Anti-Exploration with Distributional Random Network Distillation on PyTorch

This repository contains possible (not ideal one actually) PyTorch implementation of offline [SAC DRND](https://arxiv.org/abs/2401.09750) with the [wandb](https://wandb.ai/zzmtsvv/sac_drnd?workspace=user-zzmtsvv) integration. Actually, It is just a slightly modified [my realization](https://github.com/zzmtsvv/sac_rnd) of [SAC RND](https://arxiv.org/abs/2301.13616).

if you want to train the model, setup `drnd_config` in `config.py`, initialize `SACDRNDTrainer` in `trainer.py` and run its `train` method:

```python3
from trainer import SACDRNDTrainer

trainer = SACDRNDTrainer()
trainer.train()
```
if you find any bugs and mistakes in the code, please contact me :)

