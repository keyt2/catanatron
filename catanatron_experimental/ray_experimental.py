import ray
from ray import tune
from ray.rllib.examples.env.parametric_actions_cartpole import ParametricActionsCartPole
from ray.rllib.examples.models.parametric_actions_model import (
    TorchParametricActionsModel,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env

config = dict(
    {
        "env": "catanatron_gym:catanatron-v0",
        "model": {
            "custom_model": "pa_model",
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 1,
        "num_workers": 1,
        "framework": "torch",
        "hiddens" : [],
        "dueling" : False,
    },
)

