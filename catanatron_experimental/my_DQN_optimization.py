import math
from collections import namedtuple
from itertools import count
import random

import gym
import numpy as np
import optuna
from optuna.trial import TrialState

from DQN_class import DQN, ReplayMemory
from catanatron import RandomPlayer, Color
import torch
import torch.nn as nn
import torch.optim as optim


env = gym.make(
    "catanatron_gym:catanatron-v0",
    config={
        "enemies": [RandomPlayer(Color.RED)],
    },
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# def define_model(trial):
#     We optimize the number of layers, hidden units and dropout ratio in each layer.
    # n_layers = trial.suggest_int("n_layers", 1, 4)
    # layers = []
    # in_features = env.action_space.n
    # state = env.reset()
    # out_features = len(state)
    # for i in range(n_layers):
    #     out_features = trial.suggest_int("n_units_l{}".format(i), 64, 256)
    #     layers.append(nn.Linear(in_features, out_features))
    #     layers.append(nn.ReLU())
    #     in_features = out_features
    # layers.append(nn.Linear(in_features, out_features))
    # layers.append(nn.ReLU())
    #
    # return nn.Sequential(*layers)

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state = env.reset()
n_observations = len(state)


def objective(trial):
    # Generate the model.
    policy_net = DQN(n_observations, n_actions).to("cpu")
    target_net = DQN(n_observations, n_actions).to("cpu")
    memory = ReplayMemory(10000)
    steps_done = 0

    # Generate the optimizers.
    BATCH_SIZE = trial.suggest_int("BATCH_SIZE", 64, 256, log=True)
    GAMMA = 1.0 - trial.suggest_float("GAMMA", 0.0001, 0.1, log=True)
    EPS_START = trial.suggest_float("EPS_START", 0.7, 0.9, log=True)
    EPS_END = trial.suggest_float("EPS_END", 0.0001, 0.1, log=True)
    EPS_DECAY = trial.suggest_int("EPS_DECAY", 500, 2000, log=True)
    TAU = trial.suggest_float("TAU", 0.001, 0.01)
    LR = trial.suggest_float("LR", 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(optim, optimizer_name)(policy_net.parameters(), lr=LR)

    def select_action(env, steps_done):
        valid_actions = env.get_valid_actions()
        mask = np.zeros(env.action_space.n, dtype=np.float32)
        mask[valid_actions] = 1
        mask = np.array([bool(i) for i in mask])
        mask = torch.as_tensor(mask)
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
            return torch.tensor([[random.choice(env.get_valid_actions())]], device="cpu", dtype=torch.long)

    episode_durations = []
    episode_reward = []
    reward_mean = [0]
    duration_mean = [0]

    def optimize_model(steps_done):
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
                                                batch.next_state)), device="cpu", dtype=torch.bool)
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
        next_state_values = torch.zeros(BATCH_SIZE, device="cpu")
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
        num_episodes = 200

    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device="cpu").unsqueeze(0)
        for t in count():
            action = select_action(env, steps_done)
            observation, reward, terminated, info = env.step(action.item())
            reward = torch.tensor([reward], device="cpu")
            done = terminated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device="cpu").unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(steps_done)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                reward = float(reward.item())
                trial.report(reward, t)
                episode_durations.append(t + 1)
                episode_reward.append(reward)
                # if (i_episode + 1) % 50 == 0:
                #     torch.save(policy_net.state_dict(), "my_DQN1_" + str(i_episode + 1001) + "_steps.pth")
                break

    i = int((len(episode_reward) / 10) - 1)
    n = int(i * 10)
    m = int(10 * (i + 1))
    mean = np.array(episode_reward)
    mean = mean[n:m]
    mean = mean.mean()
    reward_mean.append(mean)
    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return np.asarray(reward_mean).mean()




if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", storage="sqlite:///db.sqlite3",  study_name="catan_dqn3")
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

