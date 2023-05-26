import random

import gym
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from catanatron import Game
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
from catanatron.models.player import RandomPlayer
from catanatron import Color
from stable_baselines3.common.evaluation import evaluate_policy
from catanatron_experimental.my_player import gracz_z_wiedzą_a_priori
from catanatron_server.utils import open_link

# Play a simple 4v4 game
players = [
    RandomPlayer(Color.RED),
    RandomPlayer(Color.BLUE),
    RandomPlayer(Color.WHITE),
    RandomPlayer(Color.ORANGE),
]
game = Game(players)
print(game.play())  # returns winning color
open_link(game)  # opens game in browser




