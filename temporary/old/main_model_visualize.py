from stable_baselines3 import DQN
from func_visualize import visualize_env
from gym_whackamole.envs import WhackAMole
import func_visualize

env = WhackAMole(render_mode="human")

model = DQN.load("dqn_sb_whackamole", env=env)

visualize_env(env)
