import stable_baselines3
from stable_baselines3 import DQN
import torch
from func_visualize import visualize_env
from gym_whackamole.envs import WhackAMole
import func_visualize
import torch

env = WhackAMole(render_mode="human")

nn_layers = [32,32,20] #This is the configuration of your neural network. Currently, we have two layers, each consisting of 64 neurons.
                    #If you want three layers with 64 neurons each, set the value to [64,64,64] and so on.

learning_rate = 0.001 #This is the step-size with which the gradient descent is carried out.
                      #Tip: Use smaller step-sizes for larger networks.
policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=nn_layers)
model = DQN("MlpPolicy", env,policy_kwargs = policy_kwargs,
            learning_rate=learning_rate,
            batch_size=40,  #for simplicity, we are not doing batch update.
            buffer_size=1000, #size of experience of replay buffer. Set to 1 as batch update is not done
            learning_starts=1, #learning starts immediately!
            target_update_interval=1, #update the target network immediately.
            train_freq=(1,"step"), #train the network at every step.
            exploration_initial_eps = 1, #initial value of random action probability
            exploration_fraction = 0.5, #fraction of entire training period over which the exploration rate is reduced
            gradient_steps = 1, #number of gradient steps
            seed = 2022, #seed for the pseudo random generators
            verbose=1) #Set verbose to 1 to observe training logs. We encourage you to set the verbose to 1.

model.save("dqn_sb_whackamole")
