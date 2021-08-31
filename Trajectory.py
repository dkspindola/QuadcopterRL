import numpy as np

import csv
import sys
import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.models.torch.fcnet

from Quadcopter import Quadcopter

from Configs import config, config_plot

from PlotUtils import plot, plot_csv
from Weights import compute_action_from_nn, compute_action_from_s2m


def trajectory_from_sim(checkpoint_dir, env_config, num_timesteps):
    ray.init()

    agent = ppo.PPOTrainer(config=config, env=Quadcopter)
    # restore checkpoint from checkpoint directory
    agent.restore(checkpoint_dir)

    # initialize environment
    env = Quadcopter(env_config)

    # initialize array where all actions from episode will be stored
    actions = np.array([])
    # initialize arrays where angles will be stored
    angles = np.array([])
    # initialize array where all observations from episode will be stored
    observations = np.array([])
    # reset environment and get first observation
    obs = env.reset()
    # initialize episode reward
    episode_reward = 0

    for counter in range(num_timesteps):
        # add observation to array of observations
        observations = np.concatenate([observations, obs])
        angles = np.append(angles, env.cur_ang)

        # compute action from observation
        # action = np.array(agent.compute_action(obs))
        action = compute_action_from_nn(agent, obs)
        # action = compute_action_from_s2m(obs)

        # calculate one step of environment based on action
        obs, reward, done, info = env.step(action)

        # add reward total episode reward
        episode_reward += reward

        # add action to array of actions
        actions = np.concatenate([actions, action])

    # plot observations and actions of episode
    plot(actions, angles, observations, env, episode_reward)

    ray.shutdown()


def trajectory_from_csv(pos_err_csv, lin_vel_err_csv, rot_mat_0_csv, rot_mat_1_csv, rot_mat_2_csv, ang_vel_err_csv,
                        action_csv, stabilizer_angles):
    plot_csv(read_csv(pos_err_csv), read_csv(lin_vel_err_csv), read_csv(rot_mat_0_csv), read_csv(rot_mat_1_csv),
             read_csv(rot_mat_2_csv), read_csv(ang_vel_err_csv), read_csv(action_csv), read_csv(stabilizer_angles))


def read_csv(csv_path):
    data = np.array([])
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_file:
            for col in row.split(','):
                try:
                    data = np.append(data, float(col))
                except ValueError:
                    continue
    return data


trajectory_from_sim("/home/davi/ray_results/PPO/PPO_Quadcopter_335d0_00000_0_2021-08-31_13-04-24/checkpoint_001075/checkpoint-1075", config_plot["env_config"], 4000)

"""
trajectory_from_csv("/home/davi/.config/cfclient/logdata/20210823T11-42-58/My_Drone-pos_err-20210823T11-43-53.csv",
                    "/home/davi/.config/cfclient/logdata/20210823T11-42-58/My_Drone-lin_vel_err-20210823T11-43-53.csv",
                    "/home/davi/.config/cfclient/logdata/20210823T11-42-58/My_Drone-rot_matrix0-20210823T11-43-54.csv",
                    "/home/davi/.config/cfclient/logdata/20210823T11-42-58/My_Drone-rot_matrix1-20210823T11-43-54.csv",
                    "/home/davi/.config/cfclient/logdata/20210823T11-42-58/My_Drone-rot_matrix2-20210823T11-43-55.csv",
                    "/home/davi/.config/cfclient/logdata/20210823T11-42-58/My_Drone-ang_vel_err-20210823T11-43-52.csv",
                    "/home/davi/.config/cfclient/logdata/20210823T11-42-58/My_Drone-action-20210823T11-43-52.csv",
                    "/home/davi/.config/cfclient/logdata/20210823T11-42-58/stabilizer-20210823T11-43-57.csv"
                    )"""
