from func_visualize import visualize_env
from gym_whackamole.envs import WhackAMole
import func_visualize

env = WhackAMole(render_mode="human")
params = env.params
print(params)
params['mole']['p_popping'] = 0.5
params['gaze']['radius'] = 100
params['mole']['max_life'] = 1000
env.set_params(params)

visualize_env(env)