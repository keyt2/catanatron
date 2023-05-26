import random
from typing import Iterable

import torch
from catanatron.game import Game
from catanatron.models.actions import Action
from catanatron.models.player import Player
from catanatron_gym.envs.catanatron_env import CatanatronEnv
import gym
from DQN_class import DQN

import ray
import ray.rllib.agents.ppo as ppo

from catanatron_experimental.cli.cli_players import register_player
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
# from stable_baselines3 import DQN
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
from catanatron import Color
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

from catanatron.models.enums import Action, ActionType
from catanatron_gym.features import (
    create_sample,
    create_sample_vector,
    get_feature_ordering,
)
from catanatron_gym.envs.catanatron_env import ACTIONS_ARRAY, ACTION_SPACE_SIZE, from_action_space, to_action_space
from catanatron_gym.board_tensor_features import (
    NUMERIC_FEATURES,
    create_board_tensor,)

from catanatron_experimental.machine_learning.players.reinforcement import FEATURES, hot_one_encode_action

from catanatron_gym.features import (
    create_sample,
    get_feature_ordering,
)

from catanatron.models.player import RandomPlayer
import torch.nn as nn

KWARGS = {
    "gamma": 0.0002791771829026324,
    "gae_lambda": 0.08304224564683957,
    "n_steps": 2 ** 8,
    "learning_rate": 0.0001850814015531029,
    "ent_coef": 0.009606527632294254,
    "max_grad_norm": 3.3739800379781446,
    "policy_kwargs": {
        "net_arch": [
        {"pi": [64], "vf": [64]}],
        "activation_fn": nn.Tanh,
        "ortho_init": True,
    },
}

class MyPlayer(Player):
    def decide(self, game: Game, playable_actions: Iterable[Action]):
        """Should return one of the playable_actions.

        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        # ===== YOUR CODE HERE =====
        # As an example we simply return the first action:
        print('a')
        return playable_actions[0]
        # ===== END YOUR CODE =====


# @register_player("RAY")
# class rayPlayer(Player):
#     def __init__(self):
#         super(rayPlayer, self).__init__(color)
#        self.model_path = "C:\\Users\\KOMP\\ray_results\\PPOTrainer_CatanatronEnv_2023-03-06_14-25-02xez6fw_r\\checkpoint_000101"
    #
    # config = {
    #     "env": CatanatronEnv,
    #     "num_workers": 4,
    #     "model": {"custom_model": "kp_mask"},
    # }
    # env = gym.make(
    #     "catanatron_gym:catanatron-v0",
    #     config={
    #         "enemies": [RandomPlayer(Color.RED)],
    #     },
    # )
    # obs = env.reset()
    # trainer = ppo.PPOTrainer(config=config)
    # trainer.restore(
    #     "C:\\Users\\KOMP\\ray_results\\PPOTrainer_CatanatronEnv_2023-03-14_20-58-36v_b241v0\\checkpoint_000202\\checkpoint-202")
    # def decide(self, game, playable_actions):
    #     if len(playable_actions) == 1:
    #         return playable_actions[0]
    #     state = create_sample_vector(game, self.color, FEATURES)
    #     samples = []
    #     for action in playable_actions:
    #         samples.append(np.concatenate((state, hot_one_encode_action(action))))
    #     X = np.array(samples)
    #
        # czy tak się zrobi obs???? czemu to nie działa, skoro przy SB3 działa???
        # sample = create_sample(game, game.state.current_color())
        # features = get_feature_ordering(2)
        # obs = np.array([float(sample[i]) for i in features])
        # len(obs)
        #
        # i = self.trainer.compute_single_action(observation=obs)
        # print(i)
        # flag = True
        # tab = []
        # for j in range(0, len(playable_actions)):
        #     print(playable_actions[j])
            # a = to_action_space(playable_actions[j])
            # tab.append(a)
            # if a == i:
            #     flag = False
            #     print(a)
        # if flag:
        #     i = random.choice(tab)
        #     print("Zmienione: " + str(i))
        # a = from_action_space(i, playable_actions)
        #
        # return a

# @register_player("SB3")
class gracz_z_wiedzą_a_priori(Player):
    # def mask_fn(env) -> np.ndarray:
    #     valid_actions = env.get_valid_actions()
    #     mask = np.zeros(env.action_space.n, dtype=np.float32)
    #     mask[valid_actions] = 1
    #     return np.array([bool(i) for i in mask])

    env = gym.make(
        "catanatron_gym:catanatron-v0",
        config={
            "enemies": [ValueFunctionPlayer(Color.RED)],
        },
    )
    # env = ActionMasker(env, mask_fn)
    obs = env.reset()
    # model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
    model = MaskablePPO(MaskableActorCriticPolicy, env=env).load("C:\\Users\\KOMP\\Desktop\\przedmioty i "
                                                                 "projekty\\praca "
                                                                 "licencjacka\\Catan-AI-master\\catanatron"
                                                                 "\\PPO_model10_175000_steps", env=env)
    #.load("PPO_model12.zip", env=env)
    # model.set_env(env)
    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        # env = ActionMasker(self.env, self.mask_fn)
        # model = stable_baselines3.PPO.load("C:\\Users\\KOMP\\Desktop\\przedmioty_i_projekty\\praca licencjacka\\Catan-AI-master\\catanatron\\catanatron_experimental\\PPO_model\\pytorch_variables.pth")


        sample = create_sample(game, game.state.current_color())
        features = get_feature_ordering(num_players=2)
        obs = np.array([float(sample[i]) for i in features])

        valid_actions = list(map(to_action_space, game.state.playable_actions))
        mask = np.zeros(self.env.action_space.n, dtype=np.float32)
        mask[valid_actions] = 1
        mask = np.array([bool(i) for i in mask])

        i = self.model.predict(obs, action_masks=mask)[0]
        if (game.state.is_initial_build_phase):
            a = AlphaBetaPlayer(color=game.state.current_color()).decide(game,playable_actions)
            return a
        flag = True
        tab = []
        for j in range (0, len(playable_actions)):
            # print(playable_actions[j])
            a = to_action_space(playable_actions[j])
            tab.append(a)
            if a == i:
                flag = False
                # print(a)
        if flag:
            i = random.choice(tab)
            print("Zmienione: " + str(i))
        a = from_action_space(i, playable_actions)
        # print(a)
        return a

    @register_player("SB3_wypasiony")
    class sb3Player_wypasiony(Player):
        # def mask_fn(env) -> np.ndarray:
        #     valid_actions = env.get_valid_actions()
        #     mask = np.zeros(env.action_space.n, dtype=np.float32)
        #     mask[valid_actions] = 1
        #     return np.array([bool(i) for i in mask])

        env = gym.make(
            "catanatron_gym:catanatron-v0",
            config={
                "enemies": [ValueFunctionPlayer(Color.RED)],
            },
        )
        # env = ActionMasker(env, mask_fn)
        obs = env.reset()
        # model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
        model = MaskablePPO(MaskableActorCriticPolicy, env=env).load("PPO_model10_175000_steps.zip", env=env)

        # model.set_env(env)
        def decide(self, game, playable_actions):
            if len(playable_actions) == 1:
                return playable_actions[0]
            # env = ActionMasker(self.env, self.mask_fn)
            # model = stable_baselines3.PPO.load("C:\\Users\\KOMP\\Desktop\\przedmioty_i_projekty\\praca licencjacka\\Catan-AI-master\\catanatron\\catanatron_experimental\\PPO_model\\pytorch_variables.pth")

            sample = create_sample(game, game.state.current_color())
            features = get_feature_ordering(num_players=2)
            obs = np.array([float(sample[i]) for i in features])

            valid_actions = list(map(to_action_space, game.state.playable_actions))
            mask = np.zeros(self.env.action_space.n, dtype=np.float32)
            mask[valid_actions] = 1
            mask = np.array([bool(i) for i in mask])

            # flag = True
            #
            # for n in range(93, 201):
            #     if n in valid_actions:
            #         i = n
            #         flag = False

            # if flag:
            #     for n in range(21, 93):
            #         if n in valid_actions:
            #             i = n
            #             flag = False

            # if flag:
            #     i = self.model.predict(obs, action_masks=mask)[0]

            # if i in range(93, 201):
            #     print("budowa")
            # elif i in range(21, 93):
            #     print("droga")
            # else:
            #     print("coś innego")

            i = self.model.predict(obs, action_masks=mask)[0]

            # print(i)
            flag = True
            tab = []
            for j in range(0, len(playable_actions)):
                # print(playable_actions[j])
                a = to_action_space(playable_actions[j])
                tab.append(a)
                if a == i:
                    flag = False
                    # print(a)
            if flag:
                i = random.choice(tab)
                print("Zmienione: " + str(i))
            a = from_action_space(i, playable_actions)
            # print(a)
            return a

@register_player("dqn")
class dqn(Player):
    def decide(self, game: Game, playable_actions: Iterable[Action]):
        sample = create_sample(game, game.state.current_color())
        features = get_feature_ordering(num_players=2)
        obs = np.array([float(sample[i]) for i in features])
        policy_net = DQN(len(obs), 290)
        policy_net.load_state_dict(torch.load("my_DQN1_700_steps.pth"))
        valid_actions = list(map(to_action_space, game.state.playable_actions))
        mask = np.zeros(290, dtype=np.float32)
        mask[valid_actions] = 1
        mask = np.array([bool(i) for i in mask])
        mask = torch.as_tensor(mask)
        obs = torch.tensor(obs, dtype=torch.float32, device="cpu").unsqueeze(0)
        # global steps_done
        # sample = random.random()
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        #                 math.exp(-1. * steps_done / EPS_DECAY)
        # steps_done += 1
        # if sample > eps_threshold:
        #     with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
        masked = torch.mul(policy_net(obs), mask)
                # return policy_net(state).max(1)[1].view(1, 1)
        return from_action_space(masked.max(1)[1].view(1, 1).item(), playable_actions)
        # else:
        #     return torch.tensor([[random.choice(playable_actions)]], device="gpu", dtype=torch.long)
"""
# @register_player("DQN")
class DQN(Player):
    # def mask_fn(env) -> np.ndarray:
    #     valid_actions = env.get_valid_actions()
    #     mask = np.zeros(env.action_space.n, dtype=np.float32)
    #     mask[valid_actions] = 1
    #     return np.array([bool(i) for i in mask])

    env = gym.make(
        "catanatron_gym:catanatron-v0",
        config={
            "enemies": [ValueFunctionPlayer(Color.RED)],
        },
    )
    # env = ActionMasker(env, mask_fn)
    obs = env.reset()
    # model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
    # model = MaskablePPO(MaskableActorCriticPolicy, env=env).load("PPO_model10_175000_steps.zip", env=env)
    model = DQN("MlpPolicy", env, verbose=1)
    #.load("PPO_model12.zip", env=env)
    # model.set_env(env)
    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        # env = ActionMasker(self.env, self.mask_fn)
        # model = stable_baselines3.PPO.load("C:\\Users\\KOMP\\Desktop\\przedmioty_i_projekty\\praca licencjacka\\Catan-AI-master\\catanatron\\catanatron_experimental\\PPO_model\\pytorch_variables.pth")


        sample = create_sample(game, game.state.current_color())
        features = get_feature_ordering(num_players=2)
        obs = np.array([float(sample[i]) for i in features])

        # valid_actions = list(map(to_action_space, game.state.playable_actions))
        # mask = np.zeros(self.env.action_space.n, dtype=np.float32)
        # mask[valid_actions] = 1
        # mask = np.array([bool(i) for i in mask])

        i = self.model.predict(obs)[0]
        # if (game.state.is_initial_build_phase):
        #     a = AlphaBetaPlayer(color=game.state.current_color()).decide(game,playable_actions)
        #     return a
        flag = True
        tab = []
        for j in range (0, len(playable_actions)):
            # print(playable_actions[j])
            a = to_action_space(playable_actions[j])
            tab.append(a)
            if a == i:
                flag = False
                # print(a)
        if flag:
            i = random.choice(tab)
            print("Zmienione: " + str(i))
        a = from_action_space(i, playable_actions)
        # print(a)
        return a
"""