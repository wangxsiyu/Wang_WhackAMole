from gym_lookamole.envs import LookAMole
from visualize import visualize
from evaluate import eval
from dqn.dqn import DQN
import torch

env = LookAMole(render_mode=None, render_fps = 20, n_frame_per_episode = 100)



dqn = DQN(env)
# dqn = torch.load('DQN_trained')
dqn.train(10)
torch.save(dqn,'DQN_trained')
# eval(env, dqn)

# venv = LookAMole(render_mode="human", render_fps = 20)
# visualize(venv, dqn)



