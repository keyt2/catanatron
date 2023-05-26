import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from catanatron import Color
from catanatron.models.player import RandomPlayer

env = gym.make(
    "catanatron_gym:catanatron-v0",
    config={
        "enemies": [RandomPlayer(Color.RED)],
    },
)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)



# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 180
GAMMA = 1 - 0.0012159227731809933
EPS_START = 0.8120494387504279
EPS_END = 0.0017294294096171072
EPS_DECAY = 1388
TAU = 0.009172080198594165
LR = 0.00037409111879577785

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
# optimizer = getattr(optim, "RMSprop")(policy_net.parameters(), lr=LR)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(env):
    valid_actions = env.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1
    mask = np.array([bool(i) for i in mask])
    mask = torch.as_tensor(mask)
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            masked = torch.mul(policy_net(state), mask)
            # return policy_net(state).max(1)[1].view(1, 1)
            return masked.max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.choice(env.get_valid_actions())]], device=device, dtype=torch.long)


episode_durations = []
episode_reward = []
reward_mean = [0]
duration_mean = [0]

def plot_rewards(show_result=False):
    # plt.figure(1)
    # rewards_t = torch.tensor(episode_reward, dtype=torch.float)
    # if show_result:
    #     plt.title('Result')
    # else:
    #     plt.clf()
    #     plt.title('Training...')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.plot(rewards_t.numpy())
    if len(episode_reward) % 10 == 0:
        plt.figure(1)
        plt.clf()
        plt.title('Nagrody')
        plt.xlabel('Epizod')
        plt.ylabel('Średnia nagroda')
        i = int((len(episode_reward) / 10) - 1)
        n = int(i*10)
        m = int(10 * (i + 1))
        mean = np.array(episode_reward)
        mean = mean[n:m]
        mean = mean.mean()
        reward_mean.append(mean)
        plt.plot(np.arange(start=0, stop=int(len(reward_mean)*10 - 1), step=10), reward_mean)
        plt.show()
    # Take 100 episode averages and plot them too
    # if len(rewards_t) >= 100:
    #     means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def plot_durations(show_result=False):
    # plt.figure(1)
    # durations_t = torch.tensor(episode_durations, dtype=torch.float)
    # if show_result:
    #     plt.title('Result')
    # else:
    #     plt.clf()
    #     plt.title('Training...')
    # plt.xlabel('Episode')
    # plt.ylabel('Duration')
    # plt.plot(durations_t.numpy())
    if len(episode_durations) % 10 == 0:
        plt.figure(1)
        plt.clf()
        plt.title('Liczba kroków')
        plt.xlabel('Epizod')
        plt.ylabel('Średnia liczba kroków w epizodzie')
        mean = np.array(episode_durations)
        i = int((len(episode_durations) / 10) - 1)
        n = int(i*10)
        m = int(10 * (i + 1))
        mean = mean[n:m]
        mean = mean.mean()
        duration_mean.append(mean)
        plt.plot(np.arange(start=0, stop=int(len(duration_mean)*10 - 1), step=10), duration_mean)
        plt.show()
    # Take 100 episode averages and plot them too
    # if len(durations_t) >= 100:
    #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 1000

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(env)
        observation, reward, terminated, info = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            reward = float(reward.item())
            if reward > 0:
                print(reward)
            episode_durations.append(t + 1)
            episode_reward.append(reward)
            plot_durations()
            plot_rewards()
            if (i_episode + 1) % 50 == 0:
                torch.save(policy_net.state_dict(), "my_DQN2_" + str(i_episode+1) + "_steps.pth")
                print(np.asarray(reward_mean).mean())
            break

print('Complete')
plot_durations(show_result=True)
plot_rewards(show_result=True)
plt.ioff()
plt.show()
print(episode_reward)
torch.save(policy_net.state_dict(), "my_DQN2_1000_steps.pth")