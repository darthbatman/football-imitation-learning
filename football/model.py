from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class Policy(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(Policy, self).__init__()
        layers = [nn.Linear(input_dim, hidden_layers[0])]
        for i, l in enumerate(hidden_layers[:-1]):
            layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)
        self.actor = nn.Linear(hidden_layers[-1], output_dim)

    def forward(self, x):
        x = self.layers(x)
        actor = self.actor(x)
        return actor
  
    def act(self, x, sample=False):
        actor = self.forward(x)
        return actor.argmax(-1, keepdims=True)