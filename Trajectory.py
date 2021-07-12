import numpy as np
import ray
import ray.rllib.agents.ppo as ppo

from Quadcopter import Quadcopter
from Config import config
from PlotUtils import plot

ray.init()

checkpoint_dir = \
    "/home/davi/ray_results/PPO/PPO_Quadcopter_58f98_00000_0_2021-07-12_13-50-26/checkpoint_000650/checkpoint-650"

agent = ppo.PPOTrainer(config=config, env=Quadcopter)
agent.restore(checkpoint_dir)

env = Quadcopter(config["env_config"])

actions = np.array([])
observations = np.array([])
obs = env.reset()
episode_reward = 0

for counter in range(2400):
    observations = np.concatenate([observations, obs])

    action = np.array(agent.compute_action(obs))
    obs, reward, done, info = env.step(action)

    episode_reward += reward

    actions = np.concatenate([actions, action])

plot(actions, observations, env, episode_reward)
ray.shutdown()
