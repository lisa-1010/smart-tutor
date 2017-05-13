# dqn.py
# @author: Lisa Wang
# @created: May 11 2017
#
#===============================================================================
# DESCRIPTION:
# Implements DQN model
# Given an input sequence of observations, predict the Q-value for each next action.
#
#===============================================================================
# CURRENT STATUS: In progress
#===============================================================================
# USAGE: import dqn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import tensorflow as tf
import tflearn
import numpy as np
import random
import utils
import dataset_utils as d_utils
import models_dict_utils
FLAGS = tf.flags.FLAGS



class DQNModel(object):

    def __init__(self, model_id, timesteps=100, load_checkpoint=False):
        print('Loading DQN RNN model...')

        tf.reset_default_graph()
        self.timesteps = timesteps
        self.model_dict = models_dict_utils.load_model_dict(model_id)
        self.graph_ops = build_tf_graph_drqn(timesteps, self.model_dict["n_hidden"], self.model_dict["n_outputdim"])


    def train(self, input_data, target_q_values, n_epoch=64, load_checkpoint=True):
        """
        TODO
        """
        pass

    def predict(self, input_data):
        """
        predicts Q-values for the given input sequence and each action. Result is an array of length n_actions.
        :param input_data: of shape (n_samples, n_timesteps, n_inputdim).

        :return:
        """
        n_samples, n_timesteps, n_inputdim = input_data.shape
        assert(n_inputdim == self.model_dict["n_inputdim"]), "input dimension of data doesn't match the model."
        n_actions = self.model_dict["n_outputdim"]
        if n_timesteps < self.timesteps:  # pad inputs and mask
            padded_input = np.zeros((n_samples, self.timesteps, n_inputdim))
            padded_input[:, :n_timesteps, :] = input_data[:, :, :]
            input_data = padded_input
        elif n_timesteps > self.timesteps: # truncate inputs and mask
            input_data = input_data[:, :self.timesteps, :]
        tf.reset_default_graph()
        return self.model.predict([input_data])


def build_drqn(n_timesteps, n_inputdim, n_hidden, n_actions):
    """

    :param n_timesteps:
    :param n_inputdim:
    :param n_hidden:
    :param n_actions equivalent to number of actions
    :return:
    """
    inputs = tf.placeholder(tf.float32, [None, n_timesteps, n_inputdim])
    net, hidden_states_1 = tflearn.lstm(net, n_hidden, return_seq=False, return_state=True, name="lstm_1")
    q_values = tflearn.fully_connected(net, n_actions, activation='linear')
    return inputs, q_values


def build_tf_graph_drqn(n_timesteps, n_hidden, n_actions):
    # Create shared deep q network
    q_inputs, q_net = build_drqn(n_timesteps, n_inputdim, n_hidden, n_actions)
    net_params = tf.trainable_variables()
    q_values = q_net

    # Create shared target network
    target_q_inputs, target_q_net = build_drqn(n_actions=n_actions)
    # the first len(netparams) items in trainable_variables correspond to the deeo q network
    target_net_params = tf.trainable_variables()[len(net_params):]
    target_q_values = target_q_net

    # Op for periodically updating target network with online network weights
    reset_target_network_params = \
        [target_net_params[i].assign(net_params[i])
         for i in range(len(target_net_params))]

    # Define cost and gradient update op
    a = tf.placeholder("float", [None, n_actions])
    y = tf.placeholder("float", [None])
    action_q_values = tf.reduce_sum(tf.multiply(q_values, a), reduction_indices=1)
    # compute td cost as mean square error of target q and predicted q
    cost = tflearn.mean_square(action_q_values, y)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
    grad_update = optimizer.minimize(cost, var_list=net_params)

    graph_ops = {"q_inputs": q_inputs,
                 "q_values": q_values,
                 "target_inputs": target_q_inputs,
                 "target_q_values": target_q_values,
                 "reset_target_network_params": reset_target_network_params,
                 "a": a,
                 "y": y,
                 "grad_update": grad_update}

    return graph_ops


class ExperienceBuffer():
    """
    For experience replay
    Adopted from https://github.com/awjuliani/DeepRL-Agents/blob/master/Deep-Recurrent-Q-Network.ipynb
    """
    def __init__(self, buffer_sz=100):
        self.buffer = []
        self.buffer_sz = buffer_sz # max size of buffer

    def add_experience(self, experience):
        if len(self.buffer) + 1 > self.buffer_sz:
            del self.buffer[0]
        self.buffer.append(experience)
        assert (len(self.buffer) <= self.buffer_sz), "buffer too big"

    def sample(self, batch_sz, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_sz)
        sampled_traces = []
        for episode in sampled_episodes:
            # choose a random point within the episode to start the trace
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampled_traces.append(episode[point:point + trace_length])
        sampled_traces = np.array(sampled_traces)
        return sampled_traces

