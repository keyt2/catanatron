import gym
from gym.spaces import Discrete, Box
from ray import tune
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork

import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib import agents
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf
from gym import spaces
import tensorflow as tf

from catanatron_gym.envs.catanatron_env import CatanatronEnv

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
from catanatron import Color

config = {
    "env": CatanatronEnv,
    "num_workers": 4,
    # "model": {"custom_model": "kp_mask"},
}

env = gym.make(
    "catanatron_gym:catanatron-v0",
    config={
        "enemies": [AlphaBetaPlayer(Color.RED), AlphaBetaPlayer(Color.ORANGE), AlphaBetaPlayer(Color.WHITE)],
    },
)

ray.init()
trainer = ppo.PPOTrainer(config=config)
trainer.restore("C:\\Users\\KOMP\\ray_results\\PPOTrainer_CatanatronEnv_2023-03-06_14-25-02xez6fw_r\\checkpoint_000101\\checkpoint-101")
episode_reward = 0
terminated = truncated = False

obs = env.reset()

while not terminated and not truncated:
    action = trainer.compute_single_action(obs)
    obs, reward, terminated, info = env.step(action)
    episode_reward += reward
    print(action)