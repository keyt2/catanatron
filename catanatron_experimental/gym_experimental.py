import gym
import numpy
import numpy as np
import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnRewardThreshold, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from catanatron import Color
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from catanatron.models.player import RandomPlayer
# from catanatron_experimental.my_player import sb3Player
from catanatron_gym.features import (
    create_sample,
    get_feature_ordering,
)

import tensorflow as tf
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
import torch.nn as nn
from stable_baselines3.common.monitor import Monitor
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.policies import obs_as_tensordocker-compose up

KWARGS = {
    "gamma": 0.0006997165078159262,
    "gae_lambda": 0.01261985779808512,
    "n_steps": 2 ** 3,
    "learning_rate": 9.982573563220869e-05,
    "ent_coef": 2.2775493396757354e-05,
    "max_grad_norm": 0.5194070965598586,
    # "policy_kwargs": {
    #     "net_arch": [
    #     {"pi": [64], "vf": [64]}],
    #     "activation_fn": nn.Tanh,
    #     "ortho_init": True,
    # },
}


env = gym.make(
    "catanatron_gym:catanatron-v0",
    config={
        "enemies": [RandomPlayer(Color.RED)],
    },
)

def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    # env = model.get_env()
    env = gym.make(
    "catanatron_gym:catanatron-v0",
    config={
        "enemies": [RandomPlayer(Color.RED)],
    },
)
    env = ActionMasker(env, mask_fn)
    model.set_env(env)
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, action_masks=mask_fn(env))
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward

def mask_fn(env) -> np.ndarray:
    valid_actions = env.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1

    return np.array([bool(i) for i in mask])


# Init Environment and Model
# env = gym.make("catanatron_gym:catanatron-v0")
# env = ActionMasker(env, mask_fn(env))  # Wrap to enable masking
# env = Monitor(env)
# env = VecNormalize(env)

# model = MaskablePPO(MaskableActorCriticPolicy, env=env, verbose=1, tensorboard_log="./ppo_tensorboard/", device="cuda", policy_kwargs = {"net_arch": [
#     {"pi": [256, 128, 64, 32], "vf": [256, 128, 64, 32]}],
#     "activation_fn": nn.Tanh,
#     "ortho_init": True})
# model = MaskablePPO(MaskableActorCriticPolicy, env=env).load(path="PPO_model10_175000_steps.zip", verbose=1, tensorboard_log="./ppo_tensorboard/")
model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="./ppo_tensorboard/", buffer_size=0, device="cuda")
print(model.q_net.q_net[4])
print(type(model.get_env()))




# def predict_proba(model, state):
#     obs = model.policy.obs_to_tensor(state)[0]
#     dis = model.policy.get_distribution(obs)
#     probs = dis.distribution.probs
#     probs_np = probs.detach().numpy()
#     return probs_np



# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True, use_masking=True)
# print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

checkpoint_callback = CheckpointCallback(
  save_freq=25000,
  save_path="./logs/",
  name_prefix="DQN1",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

class EveryStepCallback(BaseCallback):
    def _on_step(self) -> bool:
        sample = create_sample(env.game, env.game.state.current_color())
        features = get_feature_ordering(num_players=2)
        obs = np.array([float(sample[i]) for i in features])
        obs = model.policy.obs_to_tensor(obs)
        q_values = self.model.q_net.q_net.forward(obs[0].to(torch.float32))
        outmap_min, _ = torch.min(q_values, dim=1, keepdim=True)
        outmap_max, _ = torch.max(q_values, dim=1, keepdim=True)
        q_values = (q_values - outmap_min) / (outmap_max - outmap_min)
        # print(q_values)
        mask = np.zeros(env.action_space.n, dtype=np.float32)
        mask[env.get_valid_actions()] = 1
        mask = torch.as_tensor(mask)
        masked = torch.mul(q_values, mask)
        # print(masked)
        action = torch.argmax(masked, dim=1).reshape(-1)
        print(action)
        self.training_env.step(action)
        model.q_net_target
        return True

every_step_callback = EveryStepCallback()

callback_best = StopTrainingOnRewardThreshold(0.9, 1)
eval_callback = EvalCallback(env, callback_on_new_best=callback_best, verbose=1)

# Train
model = model.learn(total_timesteps=2000000, callback=every_step_callback)

mean_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1, deterministic=False)
print(mean_reward)



# evaluate(model)


# Save
model.save("DQN1")