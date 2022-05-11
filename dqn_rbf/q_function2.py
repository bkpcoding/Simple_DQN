import sys
from turtle import forward
sys.path.append('/home/sagar/inria/code/dqn_rbf')
from random import gauss
import exputils as eu
import torch
import math
import torch.nn as nn
import torch.functional as F
import numpy as np
from dqn_rbf.rbf_layer import RBFLayer
from torch.nn import init
import copy


class Q_Net_2(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_sizes):
        super().__init__()
        prev_size = input_dims
        count = 0
        self._layers = nn.ModuleList()
        for size in hidden_sizes:
            count += 1
            hidden_layers = nn.Sequential()
            linear_layer = nn.Linear(prev_size, size)
            nn.init.xavier_normal_(linear_layer.weight)
            nn.init.zeros_(linear_layer.bias)
            hidden_layers.add_module('linear', linear_layer)
            hidden_layers.add_module('nonLinearity', nn.ReLU())
            prev_size = size
        self.output_layer = nn.Sequential()
        linear_layer = nn.Linear(prev_size, output_dims)
        nn.init.xavier_normal_(linear_layer.weight)
        nn.init.zeros_(linear_layer.bias)
        self.output_layer.add_module('linear', linear_layer)


    def forward(self, input):
        if isinstance(input, np.ndarray):
            x = torch.from_numpy(input)
        else:
            x = input
        for layer in self._layers:
            x = layer(x)
        return self.output_layer(x)
            

                
class Q_Net_3(nn.Module):
    def __init__(self, state_size, action_size):
        super(Q_Net_3, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, input):
        if isinstance(input, np.ndarray):
            x = torch.from_numpy(input)
        else:
            x = input
        return self.fc(x)

