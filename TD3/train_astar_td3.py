import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from numpy import inf

from replay_buffer import ReplayBuffer
from astar_env import GazeboEnv
import logging


# Setup logging for better tracking
logging.basicConfig(level=logging.INFO)


def evaluate(network, epoch, eval_episodes=10):
    """Evaluate the performance of the network."""
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, _ = env.step(a_in)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    logging.info(f"Average Reward over {eval_episodes} Evaluation Episodes, Epoch {epoch}: {avg_reward:.4f}, Collisions: {avg_col:.4f}")
    return avg_reward


class Actor(nn.Module):
    def __init__(self, augmented_state, action_dim):
        """Initialize the Actor network."""
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(augmented_state, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        """Forward pass through the actor network."""
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    def __init__(self, augmented_state, action_dim):
        """Initialize the Critic network."""
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(augmented_state, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(augmented_state, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        """Forward pass through the critic network."""
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2


class OccupancyGridMap:
    def __init__(self, grid_size, resolution):
        """Initialize the occupancy grid map."""
        self.grid_size = grid_size
        self.resolution = resolution
        self.grid = np.zeros((grid_size, grid_size))
        self.previous_scans = []

    def update_map(self, lidar_scan, robot_position):
        """Update the occupancy map using LIDAR scan data."""
        self.grid.fill(0)

        for i, distance in enumerate(lidar_scan):
            angle = robot_position[2] + i * (2 * np.pi / len(lidar_scan))
            x = robot_position[0] + distance * np.cos(angle)
            y = robot_position[1] + distance * np.sin(angle)

            grid_x = int((x / self.resolution) + self.grid_size / 2)
            grid_y = int((y / self.resolution) + self.grid_size / 2)

            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.grid[grid_x, grid_y] = 1

        self.classify_obstacles(lidar_scan)

    def display(self):
        """Display the occupancy grid map."""
        plt.imshow(self.grid, cmap="gray")
        plt.title("Occupancy Grid Map")
        plt.show()

    def classify_obstacles(self, current_scan):
        """Classify dynamic and static obstacles."""
        if not self.previous_scans:
            self.previous_scans.append(current_scan)
            return

        differences = np.abs(np.array(current_scan) - np.array(self.previous_scans[-1]))
        threshold = 0.2
        for i, diff in enumerate(differences):
            if diff > threshold:
                logging.info(f"Dynamic obstacle detected at angle {i}")

        self.previous_scans.append(current_scan)
        if len(self.previous_scans) > 5:
            self.previous_scans.pop(0)


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        """Initialize the TD3 network."""
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, state):
        """Return the action chosen by the actor."""
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        """Train the TD3 network."""
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        for it in range(iterations):
            batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states = replay_buffer.sample_batch(batch_size)

            batch_states = np.array(batch_states, dtype=np.float32)
            batch_actions = np.array(batch_actions, dtype=np.float32)
            batch_rewards = np.array(batch_rewards, dtype=np.float32)
            batch_dones = np.array(batch_dones, dtype=np.float32)
            batch_next_states = np.array(batch_next_states, dtype=np.float32)

            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            next_action = self.actor_target(next_state)

            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))

            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            current_Q1, current_Q2 = self.critic(state, action)
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            av_loss += loss
        self.iter_count += 1
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Initialize environment and other parameters
seed = 0
eval_freq = 5000
max_ep = 500
eval_ep = 10
max_timesteps = 5e6
expl_noise = 1
expl_decay_steps = 5000
expl_min = 0.1
batch_size = 128
discount = 0.99
tau = 0.005
policy_noise = 0.1
noise_clip = 0.4
policy_freq = 2
buffer_size = 1e6
file_name = "TD3_velodyne"
save_model = True
load_model = False
random_near_obstacle = False

# Create the network storage folders
os.makedirs("./results", exist_ok=True)
os.makedirs("./pytorch_models", exist_ok=True)

# Initialize the environment
environment_dim = 136
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1

# Initialize the TD3 network
network = TD3(state_dim, action_dim, max_action)

# Create the replay buffer
replay_buffer = ReplayBuffer(buffer_size, seed)
if load_model:
    try:
        network.load(file_name, "./pytorch_models")
    except:
        logging.warning("Could not load the stored model parameters, initializing training with random parameters")

# Create evaluation data store
evaluations = []

# Training loop
timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1
count_rand_actions = 0
random_action = []
occupancy_map = OccupancyGridMap(grid_size=10, resolution=0.5)

while timestep < max_timesteps:
    if done:
        if timestep != 0:
            network.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

        if timesteps_since_eval >= eval_freq:
            evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
            network.save(file_name, directory="./pytorch_models")
            np.save(f"./results/{file_name}", evaluations)
            epoch += 1

        state = env.reset()
        done = False
        lidar_scan = state[4:-8]
        robot_position = [state[0], state[1], state[2]]
        occupancy_map.update_map(lidar_scan, robot_position)

        map_flat = occupancy_map.grid.flatten()
        augmented_state = np.concatenate((state, map_flat)).astype(np.float32)

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    if expl_noise > expl_min:
        expl_noise -= (1 - expl_min) / expl_decay_steps

    action = network.get_action(np.array(augmented_state))
    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(-max_action, max_action)

    if random_near_obstacle:
        if np.random.uniform(0, 1) > 0.85 and np.min(lidar_scan) < 0.6 and count_rand_actions < 1:
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.random.uniform(-1, 1, 2)

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action
            action[0] = -1

    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    episode_reward += reward

    replay_buffer.add(state, action, reward, done_bool, next_state)

    state = next_state
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    network.save(file_name, directory="./models")
np.save(f"./results/{file_name}", evaluations)
