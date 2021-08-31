import sys
import ray
import ray.rllib.agents.ppo as ppo
import numpy as np
from numpy import tanh, exp
import NN_S2M
from NN_S2M import *

from Quadcopter import Quadcopter
from Configs import config


def compute_action_from_nn(model, observation):
    weights_dict = model.get_policy().get_weights()

    hidden_weights0 = weights_dict["_hidden_layers.0._model.0.weight"]
    hidden_biases0 = weights_dict["_hidden_layers.0._model.0.bias"]
    hidden_weights1 = weights_dict["_hidden_layers.1._model.0.weight"]
    hidden_biases1 = weights_dict["_hidden_layers.1._model.0.bias"]
    output_weights = weights_dict["_logits._model.0.weight"]
    output_biases = weights_dict["_logits._model.0.bias"]

    hidden_layer0 = tanh(np.dot(hidden_weights0, observation) + hidden_biases0)
    hidden_layer1 = tanh(np.dot(hidden_weights1, hidden_layer0) + hidden_biases1)
    output_layer = np.dot(output_weights, hidden_layer1) + output_biases
    # actions = output_layer[:4] + exp(output_layer[4:8]) * np.random.normal(size=(4,))
    actions = output_layer[:4]

    return actions


def compute_action_from_s2m(observation):
    hidden_layer0 = tanh(np.dot(observation, layer_0_weight) + layer_0_bias)
    hidden_layer1 = tanh(np.dot(hidden_layer0, layer_1_weight) + layer_1_bias)
    output_layer = np.dot(hidden_layer1, layer_2_weight) + layer_2_bias
    # actions = output_layer[:4] + exp(output_layer[4:8]) * np.random.normal(size=(4,))
    actions = output_layer[:4]
    return actions


def make_header(checkpoint_dir):
    ray.init()

    agent = ppo.PPOTrainer(config=config, env=Quadcopter)
    # restore checkpoint from checkpoint directory
    agent.restore(checkpoint_dir)

    np.set_printoptions(threshold=sys.maxsize)

    weights_dict = agent.get_policy().get_weights()

    hidden_weights0 = weights_dict["_hidden_layers.0._model.0.weight"]
    hidden_biases0 = weights_dict["_hidden_layers.0._model.0.bias"]
    hidden_weights1 = weights_dict["_hidden_layers.1._model.0.weight"]
    hidden_biases1 = weights_dict["_hidden_layers.1._model.0.bias"]
    output_weights = weights_dict["_logits._model.0.weight"]
    output_biases = weights_dict["_logits._model.0.bias"]

    weights_dict_str = "// Checkpoint: " + checkpoint_dir.replace("/home/davi/ray_results", "..") + "\n"
    weights_dict_str += "#ifndef BRAIN_H\n#define BRAIN_H\n" \
                   "const int structure[3][2] = {{64, 18}, {64, 64}, {8, 64}};\n"
    weights_dict_str += "const float HIDDEN_WEIGHTS_0[64][18] = " + \
                    (((repr(hidden_weights0).replace("[", "{")).replace("]", "}")).replace("\n", "")).replace(" ", "") + ";\n"
    weights_dict_str += "const float HIDDEN_BIASES_0[64] = " + \
                    (((repr(hidden_biases0).replace("[", "{")).replace("]", "}")).replace("\n", "")).replace(" ", "") + ";\n"
    weights_dict_str += "const float HIDDEN_WEIGHTS_1[64][64] = " + \
                    (((repr(hidden_weights1).replace("[", "{")).replace("]", "}")).replace("\n", "")).replace(" ", "") + ";\n"
    weights_dict_str += "const float HIDDEN_BIASES_1[64] = " + \
                    (((repr(hidden_biases1).replace("[", "{")).replace("]", "}")).replace("\n", "")).replace(" ", "") + ";\n"
    weights_dict_str += "const float OUTPUT_WEIGHTS[8][64] = " + \
                    (((repr(output_weights).replace("[", "{")).replace("]", "}")).replace("\n", "")).replace(" ", "") + ";\n"
    weights_dict_str += "const float OUTPUT_BIASES[8] = " + \
                    (((repr(output_biases).replace("[", "{")).replace("]", "}")).replace("\n", "")).replace(" ", "") + ";\n"
    weights_dict_str += "#endif\n"

    weights_dict_str = weights_dict_str.replace("array", "")
    weights_dict_str = weights_dict_str.replace(",dtype=float32", "")
    weights_dict_str = weights_dict_str.replace("(", "")
    weights_dict_str = weights_dict_str.replace(")", "")

    f = open("brain.h", "w")
    f.write(weights_dict_str)
    f.close()

    ray.shutdown()


# make_header('/home/davi/ray_results/PPO/PPO_Quadcopter_335d0_00000_0_2021-08-31_13-04-24/checkpoint_000980/checkpoint-980')
