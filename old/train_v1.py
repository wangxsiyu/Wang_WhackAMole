from stable_baselines3 import DQN
from func_visualize import visualize_env
from gym_whackamole.envs import WhackAMole
import func_visualize

env = WhackAMole(render_mode=None)
params = env.params
params['mole']['is_fixed_location'] = 1
params['gaze']['radius'] = 100
model = DQN.load("dqn_sb_whackamole", env=env)
model.learn(1000000, log_interval= 10)
model.save('dqn_sb_whackamole')
print('done')

env.setup_rendermode("human")
visualize_env(env, model)
