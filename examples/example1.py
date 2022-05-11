import sys
sys.path.append('/home/sagar/inria/code/dqn_rbf')

import dqn_rbf
from dqn_rbf.dqn import DQN
import gym


env = gym.make('CartPole-v0')
dqn_rbf.run_training_and_test()