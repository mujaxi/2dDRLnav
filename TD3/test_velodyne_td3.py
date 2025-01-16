import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from velodyne_env import GazeboEnv

# export ROS_MASTER_URI=http://10.9.17.52:11311
# export ROS_HOSTNAME=10.9.17.52


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        # Function to load network parameters
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )


# Set the parameters for the implementation
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
device = torch.device("cpu")
seed = 0  # Random seed number
max_ep = 5e6  # maximum number of steps per episode
file_name = "TD3_velodyne"  # name of the file to load the policy from


# Create the testing environment
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(1)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2

# Create the network
network = TD3(state_dim, action_dim)
try:
    network.load(file_name, "./pytorch_models")
except:
    raise ValueError("Could not load the stored model parameters")

done = False
episode_timesteps = 0
state = env.reset()
# time.sleep(5)
# Begin the testing loop
while True:
    action = network.get_action(np.array(state))
    # print (state)

    action [0] = ((action[0] + 1) / 2) * 0.26
    a_in = [action[0], action[1]]
    next_state, reward, done, target = env.step(a_in)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)

    # On termination of episode
    if done:
        env.save_trajectory()
        print("Done!")
        state = env.reset() #for training
        # time.sleep(1)
        done = False
        episode_timesteps = 0
        
        
    else:
        state = next_state
        episode_timesteps += 1


