from email.policy import default
from glob import glob
from pyexpat import model
import exputils as eu
import exputils.data.logging as log
import dqn_rbf
import torch
import torch.nn.functional as F
import warnings
import numpy as np
import gym
import torch.nn as nn

from dqn_rbf.dqn import DQN
from dqn_rbf.q_function import Q_net
from dqn_rbf.rbf_layer import RBFLayer
def run_training_and_test(config = None, **kwargs):
    default_config = eu.AttrDict(
            agent = eu.AttrDict(
                cls= DQN,
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
                device = 'cuda',
                gamma = 0.99,
                epsilon = 0.01,
                sync_freq = 5,
                optimizer = eu.AttrDict(
                    cls = torch.optim.Adam,
                    lr = 1e-3,
                ),
            ),
            
            episodes = 10000,
            batch_size = 16,
            iterations = 100000,
            size_replay = 100000,
            )
    config  = eu.combine_dicts(kwargs, config, default_config, copy_mode= 'copy')
    losses = []
    agent = eu.misc.create_object_from_config(config.agent)
    agent.fill_experience(config.size_replay)
    for _ in range(config.iterations):
        losses.append(agent.train(config.batch_size))
    log.add_value("agent_loss", np.mean(losses))
    print(np.mean(losses))
    test(agent, agent.env, config.episodes)
    print("*************testing random agent**********")
    random_agent(agent.env, config)
    
def test(agent, env, episodes):
    state = env.reset()
    rewards_list = []
    for e in range(episodes):
        state = env.reset()
        rewards = 0
        Done = False
        while not Done:
            action = agent.get_action(state)
            next_state, reward, Done, _ = env.step(action)
            rewards += reward
            state = next_state
        rewards_list.append(rewards)
        if (e%100) == 0:
            log.add_value("Average_reward over last 100 episodes", np.mean(rewards_list))
            print("Average_reward over last 100 episodes", np.mean(rewards_list))
            rewards_list.clear()
    env.close()

def random_agent(env, config):
    state = env.reset()
    random_agent = eu.misc.create_object_from_config(config.agent)
    rewards_list = []
    for e in range(10000):
        state = env.reset()
        rewards = 0
        Done = False
        while not Done:
            action = random_agent.get_action(state)
            next_state, reward, Done, _ = env.step(action)
            rewards += reward
            state = next_state
        rewards_list.append(rewards)
        if (e%100) == 0:
            log.add_value("Average_reward over last 100 episodes", np.mean(rewards_list))
            print("Average_reward over last 100 episodes", np.mean(rewards_list))

            rewards_list.clear()


