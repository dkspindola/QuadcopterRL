import numpy as np
import ray
import ray.rllib.agents.ppo as ppo

from Quadcopter import Quadcopter
from Config import config
from PlotUtils import plot

ray.init()

checkpoint_dir = \
    "/home/davi/ray_results/PPO/PPO_Quadcopter_ae552_00000_0_2021-07-01_15-46-05/checkpoint_001300/checkpoint-1300"

agent = ppo.PPOTrainer(config=config, env=Quadcopter)
agent.restore(checkpoint_dir)

env = Quadcopter(config["env_config"])

actions = np.array([])
observations = np.array([])

episode_reward = 0
obs = env.reset()
for counter in range(4000):
    action = np.array(agent.compute_action(obs))
    obs, reward, done, info = env.step(action)

    episode_reward += reward

    actions = np.concatenate([actions, action])
    observations = np.concatenate([observations, obs])

plot(actions, observations, env)
ray.shutdown()
