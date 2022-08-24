# -*- coding: utf-8 -*-
"""
@author: Mohammad Asim
"""
from dataclasses import dataclass, asdict, field

@dataclass
class Settings:
    name: str
    lr: float=0.0008
    lr_decay: float=0.8
    patience: int=12
    epochs: int=1000
    alpha: float=0.01 # Regularization parameter
    lmbda: float=0.8 # Gradient Constancy parameter
    epsilon: float=0.001 # Penalisation parameter
    multi_scale_weights: list=field(default_factory=list)
    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}