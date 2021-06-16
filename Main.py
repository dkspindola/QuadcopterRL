import ray
from ray import tune
from ray.tune import registry
from ray.rllib.utils.framework import try_import_torch

from Quadcopter import Quadcopter
from Config import config


torch, nn = try_import_torch()


if __name__ == "__main__":
    ray.init()

    registry.register_env("Quadcopter", lambda c: Quadcopter(c))

    results = tune.run("PPO", config=config)

    ray.shutdown()
