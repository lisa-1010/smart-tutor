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



class DRQNModel(object):

    def __init__(self, model_id, timesteps=100, load_checkpoint=False):
        print('Loading DQN RNN model...')

        tf.reset_default_graph()
        self.timesteps = timesteps
        self.model_dict = models_dict_utils.load_model_dict(model_id)
        self.graph_ops = build_tf_graph_drqn(timesteps, self.model_dict["n_hidden"], self.model_dict["n_outputdim"])







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
    net, hidden_states_1 = tflearn.lstm(inputs, n_hidden, return_seq=True, return_state=True, name="lstm_1")
    q_values = tflearn.lstm(net, n_actions, activation='linear', name="lstm_2")
    q_values = tf.stack(q_values, axis=1) # tensor shape(None, n_timesteps, n_actions)
    return inputs, q_values


def build_tf_graph_drqn(n_timesteps, n_inputdim, n_hidden, n_actions):
    # Create shared deep q network
    q_inputs, q_net = build_drqn(n_timesteps, n_inputdim, n_hidden, n_actions)
    net_params = tf.trainable_variables()
    q_values = q_net

    # TODO: for now we will use a single Q network. Since we use offline data, there is less of a risk to instability.
    # # Create shared target network
    # target_q_inputs, target_q_net = build_drqn(n_actions=n_actions)
    # # the first len(netparams) items in trainable_variables correspond to the deeo q network
    # target_net_params = tf.trainable_variables()[len(net_params):]
    # target_q_values = target_q_net
    #
    # # Op for periodically updating target network with online network weights
    # reset_target_network_params = \
    #     [target_net_params[i].assign(net_params[i])
    #      for i in range(len(target_net_params))]

    # Define cost and gradient update op
    a = tf.placeholder("float", [None, n_timesteps, n_actions])
    y = tf.placeholder("float", [None, n_timesteps])
    action_q_values = tf.reduce_sum(tf.multiply(q_values, a), reduction_indices=1)  # shape [None, n_timesteps]
    # compute td cost as mean square error of target q and predicted q
    cost = tflearn.mean_square(action_q_values, y)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
    grad_update = optimizer.minimize(cost, var_list=net_params)

    # graph_ops = {"q_inputs": q_inputs,
    #              "q_values": q_values,
    #              "target_inputs": target_q_inputs,
    #              "target_q_values": target_q_values,
    #              "reset_target_network_params": reset_target_network_params,
    #              "a": a,
    #              "y": y,
    #              "grad_update": grad_update}

    graph_ops = {"q_inputs": q_inputs,
                 "q_values": q_values,
                 "a": a,
                 "y": y,
                 "grad_update": grad_update}
    return graph_ops



def train(graph_ops, gamma=0.99, batch_sz=16, n_epoch=16, target_update_freq=10, load_checkpoint=True, ckpt_path=""):
    """
    Treat our offline data as the experience replay buffer and we only train on "experience"
    Data could be provided with the experience buffer (list of (s,a,r,s') tuples)
    1. experience buffer could go through data in order, in chunks of 64
    2. randomly sample a batch of 64 samples
    """

    # add "Experiences" from our offline data to the experience buffer
    #
    tf.reset_default_graph()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=5)
    experience_buffer = ExperienceBuffer()

    # unpack graph_ops

    q_inputs = graph_ops["q_inputs"]
    q_values = graph_ops["q_values"]

    a = graph_ops["a"]
    y = graph_ops["y"]
    grad_update = graph_ops["grad_update"]

    with tf.Session() as sess:
        if load_checkpoint == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        sess.run(init)

        training_steps = n_epoch * (experience_buffer.buffer_sz / batch_sz)
        # one training step corresponds to one update to the Q network

        for i in xrange(training_steps):
            # traces is a list/batch of experience traces. Each trace is a tuple of state_action_Data and rewards.
            train_batch = experience_buffer.sample_in_order(batch_sz)
            print(train_batch.shape)

            s_batch =  np.vstack(train_batch[:,0]) # current states
            a_batch = np.vstack(train_batch[:,1]) # actions

            q_inputs_batch = combine_states_actions(s_batch, a_batch    )
            r_batch = np.vstack(train_batch[:,2]) # rewards
            sp_batch = np.vstack(train_batch[:, 3])  # next states
            Q = q_values.eval(session=sess, feed_dict={q_inputs: [s_batch]})

            y_batch = r_batch + gamma * q_values.eval(session=sess, feed_dict={q_inputs: [sp_batch]})

            # Update the network with our target values
            sess.run(grad_update, feed_dict={y: y_batch,
                                                a: a_batch,
                                                q_inputs: q_inputs_batch})



def load_data_into_experience_buffer(self, data):
    """

    Loads offline data from data generator including state, action, rewards into self.experience_buffer.
    :param data:
    :return:
    """
    # TODO: Complete this function

    n_samples = data.shape[0]
    self.experience_buffer = ExperienceBuffer(buffer_sz=n_samples)
    for i in xrange(n_samples):
        self.experience_buffer.add_episode((s,a,r))





class ExperienceBuffer(object):
    """
    For experience replay
    based on implementation from https://github.com/awjuliani/DeepRL-Agents/blob/master/Deep-Recurrent-Q-Network.ipynb
    Extended with sample_in_order.
    """
    def __init__(self, buffer_sz=100):
        self.buffer = []
        self.buffer_sz = buffer_sz # max size of buffer
        self.cur_episode_index = 0
        self.max_episode_length = 0

    def add_episode(self, episode):
        if len(self.buffer) + 1 > self.buffer_sz:
            del self.buffer[0]
        self.buffer.append(episode)
        self.max_episode_length = max(self.max_episode_length, len(episode))
        assert (len(self.buffer) <= self.buffer_sz), "buffer too big"

    def sample(self, batch_sz, trace_length=-1):
        sampled_episodes = random.sample(self.buffer, batch_sz)
        return self._get_traces_from_episodes(sampled_episodes, trace_length)

    def sample_in_order(self, batch_sz, trace_length=-1):
        """
        This function can be used to go through all episodes in experience buffer in order, e.g. if you want to make sure
        that all episodes have been seen.
        Keeps track of the index of the next episode. If samples go over end of the buffer, it wraps around.
        :param batch_sz: number of traces to return
        :param trace_length: max length of each trace
        :return:
        """
        assert (batch_sz <= len(self.buffer)), "batch size is larger than experience buffer. "
        sampled_episodes = []
        if self.cur_episode_index + batch_sz <= len(self.buffer):
            sampled_episodes = self.buffer[self.cur_episode_index:self.cur_episode_index + batch_sz]
            self.cur_episode_index += batch_sz
        else:
            overhang = self.cur_episode_index + batch_sz - len(self.buffer) # number of samples to wrap around
            sampled_episodes = self.buffer[self.cur_episode_index:] + self.buffer[0:overhang]
            self.cur_episode_index = overhang
        return self._get_traces_from_episodes(sampled_episodes, trace_length)


    def _get_traces_from_episodes(self, sampled_episodes, trace_length=-1):
        """

        :param sampled_episodes:
        :param trace_length:  if -1, then return full episodes.
        :return:
        """
        sampled_traces = []
        if trace_length != -1:
            for episode in sampled_episodes:
            # choose a random point within the episode to start the trace
                point = np.random.randint(0, len(episode) + 1 - trace_length)
                sampled_traces.append(episode[point:point + trace_length])
        else:
            sampled_traces = sampled_episodes
        sampled_traces = np.array(sampled_traces)
        return sampled_traces





