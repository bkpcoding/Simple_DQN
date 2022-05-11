import sys
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

class NonLinearity(nn.Module):
    """Wrapper class for non linear function or module.
    Args:
        non_linear (callable or type): Non-linear function or type to be
            wrapped.
    """

    def __init__(self, non_linear):
        super().__init__()

        if isinstance(non_linear, type):
            self.module = non_linear()
        elif callable(non_linear):
            self.module = copy.deepcopy(non_linear)
        else:
            raise ValueError(
                'Non linear function {} is not supported'.format(non_linear))


class Q_net(nn.Module):

    @staticmethod
    def default_config():
        return eu.AttrDict(
                    RBF = eu.AttrDict(
                        cls = RBFLayer,
                        n_neurons_per_input = 5,
                        ranges = [-1,1],
                        sigma = 1,
                        is_trainable = True,
                    ),
                    rbf_on = True,
                    input_dims = 4,
                    output_dims = 2,
                    hidden_sizes = [64, 128],
                    hidden_nonlinearity = torch.relu,
                    hidden_w_init = nn.init.xavier_normal_,
                    hidden_b_init = nn.init.zeros_,
                    output_nonlinearities = None,
                    output_w_inits = nn.init.xavier_normal,
                    output_b_inits = nn.init.zeros_,
                    layer_normalization = False

        )
    
    def __init__(self, config = None, **kwargs):
        super().__init__()
        self.config = eu.combine_dicts(kwargs, config, self.default_config())
        self._layers = nn.ModuleList()
        prev_size = self.config.input_dims
        if self.config.rbf_on:
            rbf_layer = eu.misc.create_object_from_config(self.config.RBF, self.config.input_dims)
            self._layers.append(rbf_layer)
            prev_size = rbf_layer.n_out
            print("using rbf")
        for size in self.config.hidden_sizes:
            hidden_layers = nn.Sequential()
            if self.config.layer_normalization:
                hidden_layers.add_module('layer_normalization',
                                          nn.LayerNorm(prev_size))
            linear_layer = nn.Linear(prev_size, size)
            self.config.hidden_w_init(linear_layer.weight)
            self.config.hidden_b_init(linear_layer.bias)
            hidden_layers.add_module('linear', linear_layer)
            if self.config.hidden_nonlinearity:
                non_linearity = nn.ReLU()
                hidden_layers.add_module('non_linearity',
                                         non_linearity)
            self._layers.append(hidden_layers)
            prev_size = size

        self.output_layer = nn.Sequential()
        linear_layer = nn.Linear(prev_size, self.config.output_dims)
        self.config.output_w_inits(linear_layer.weight)
        self.config.output_b_inits(linear_layer.bias)
        self.output_layer.add_module('linear', linear_layer)
        """
        Still have to add output non-linearities
        """

    def forward(self, input):
        if isinstance(input, np.ndarray):
            x = torch.from_numpy(input)
        else:
            x = input
        x = x.unsqueeze(0)
        for layer in self._layers:
            x = layer(x)
        return self.output_layer(x)
    

        



