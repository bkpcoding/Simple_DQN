from multiprocessing.sharedctypes import Value
import numpy as np
import torch
import exputils as eu
import exputils.data.logging as log
import torch.nn as nn
import gym
import warnings
import random
from collections import deque
from dqn_rbf.q_function import Q_net
import copy
from dqn_rbf.q_function2 import Q_Net_2, Q_Net_3
from dqn_rbf.rbf_layer import RBFLayer

class DQN:
    """
    DQN  agent
    """
    @staticmethod
    def default_config():
        dc = eu.AttrDict(
                model = eu.AttrDict(
                    cls = Q_net,
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
                ),
            env = 'CartPole-v0',
            device='cuda', # 'cpu', 'cuda'
            gamma=0.95,
            epsilon=0.1,
            sync_freq=5,
            exp_replay_size=256,
            #batch_size=16,
            #n_iterations=4,
            #train_every_n_steps=128,

            optimizer=eu.AttrDict(
                cls=torch.optim.Adam,
                lr=1e-3,
            ),

            log_loss_per_step = True,
            log_epsilon_per_episode = True,
        )
        return dc


    def __init__(self, config = None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        if self.config.device == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
                warnings.warn('Cuda not available. Using CPU as device ...')
        else:
            self.device = torch.device(self.config.device)
        
        if isinstance(self.config.env, dict):
            if 'cls' in self.config.env:
                self.env = eu.misc.call_function_from_config(self.config.env, func_attribute_name='cls')
            else:
                self.env = eu.misc.call_function_from_config(self.config.env)
        elif isinstance(self.config.env, str):
            self.env = gym.make(self.config.env)
        else:
            self.env = self.config.env
        if not isinstance(self.env.action_space, gym.spaces.Discrete):
            raise ValueError('DQN agent works with discrete action spaces only!')



        self.input_dims = self.env.observation_space.shape[0]
        self.output_dims = self.env.action_space.n
        #self.q_net = eu.misc.create_object_from_config(self.config.model, input_dims = self.input_dims)
        #self.q_net = Q_net(rbf_on = True, input_dims = self.input_dims,
        #                    output_dims = self.output_dims, config= config)
        self.q_net = Q_Net_3(self.input_dims, self.output_dims)
        self.target_net = copy.deepcopy(self.q_net)
        self.q_net.to(self.device)
        self.target_net.to(self.device)

        self.optimizer = eu.misc.call_function_from_config(
                         self.config.optimizer,
                         self.q_net.parameters(),
                         func_attribute_name = 'cls'
        )

        self.network_sync_freq = self.config.sync_freq
        self.network_sync_counter = 0
        self.gamma = torch.tensor(self.config.gamma).float().to(self.device)
        self.experience_replay = deque(maxlen=self.config.exp_replay_size)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.episode_counter = 0
        self.steps_since_last_training = 0
        self.epsilon = self.config.epsilon
        self.loss_fn = nn.MSELoss()

    def get_action(self, state):
        with torch.no_grad():
            if np.random.rand() <= self.epsilon:
                action = self.env.action_space.sample()
            else:
                q_values = self.q_net(state)
                action = torch.argmax(q_values)
        return int(action)
    
    def collect_experience(self, experience : list):
        self.experience_replay.append(experience)
        return

    def fill_experience(self, size):
        state = self.env.reset()
        done = False
        for _ in range(size):
            action = self.get_action(state)
            new_state, reward, done, _ = self.env.step(action)
            if done == True:
                state = self.env.reset()
                self.env.reset()
            self.collect_experience([state, action, reward, new_state])
            state = new_state

    def sample_from_experience(self, sample_size):
        if (len(self.experience_replay) < sample_size):
            sample_size = len(self.experience_replay)
        sample = random.sample(self.experience_replay, sample_size)
        s = torch.tensor([exp[0] for exp in sample]).float()
        a = torch.tensor([exp[1] for exp in sample]).float()
        rn = torch.tensor([exp[2] for exp in sample]).float()
        sn = torch.tensor([exp[3] for exp in sample]).float()
        return s, a, rn, sn

    def get_q_next(self, state):
        with torch.no_grad():
            qp = self.target_net(state)
        q, _ = torch.max(qp, axis=1)
        return q


    def train(self, batch_size):
        s, a, rn, sn = self.sample_from_experience(sample_size=batch_size)
        if self.network_sync_counter == self.network_sync_freq:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.network_sync_counter = 0

        qp = self.q_net(s.to(self.device))
        pred_return, _ = torch.max(qp, axis = 1)

        q_next = self.get_q_next(sn.to(self.device))
        target_return = rn.to(self.device) + self.gamma*q_next

        loss = self.loss_fn(pred_return, target_return)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
 
        self.network_sync_counter += 1
        return loss.item()


