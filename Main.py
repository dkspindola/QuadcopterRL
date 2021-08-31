import ray
from ray import tune
from ray.tune import registry
from ray.rllib.utils.framework import try_import_torch

from Quadcopter import Quadcopter
from Configs import config


torch, nn = try_import_torch()


if __name__ == "__main__":
    ray.init(local_mode=False)

    registry.register_env("Quadcopter", lambda c: Quadcopter(c))

    results = tune.run("PPO", config=config, checkpoint_freq=5)

    ray.shutdown()
