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

    results = tune.run("PPO", config=config, checkpoint_freq=10, restore="/home/davi/ray_results/PPO/PPO_Quadcopter_2729b_00000_0_2021-07-12_11-33-02/checkpoint_000400/checkpoint-400")

    ray.shutdown()
