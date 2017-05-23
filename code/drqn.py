# drqn.py
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
# USAGE: import drqn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import tensorflow as tf
import tflearn
import numpy as np
import random

from helpers import *
import utils
import data_generator as dg
import student as st
import exercise as exer

from mcts import StudentExactSim

from experience_buffer import ExperienceBuffer

import dataset_utils as d_utils
import models_dict_utils


from joblib import Parallel, delayed

FLAGS = tf.flags.FLAGS

class DRQNModel(object):

    def __init__(self, model_id, timesteps=100, load_checkpoint=False, load_ckpt_path=""):
        print('Loading DQN RNN model...')
        tf.reset_default_graph()
        self.timesteps = timesteps
        self.model_dict = models_dict_utils.load_model_dict(model_id)
        self.graph_ops = build_tf_graph_drqn_tflearn(timesteps, self.model_dict["n_inputdim"], self.model_dict["n_hidden"], self.model_dict["n_outputdim"])
        self.experience_buffer = ExperienceBuffer()


    def predict(self, inputs):
        """
        First predicts Q-values for the given input sequence and each action.
        Then takes argmax to get best action
        Takes in a batch.
        :param input_data: of shape (n_samples, n_timesteps, n_inputdim).

        :return: best next action
        """
        tf.reset_default_graph()
        q_inputs = self.graph_ops["q_inputs"]
        q_values = self.graph_ops["q_values"]

        evaluator = tflearn.Evaluator(q_values, model=None, session=None)

        n_samples, n_timesteps, n_inputdim = inputs.shape
        assert(n_inputdim == self.model_dict["n_inputdim"]), "input dimension of data doesn't match the model."

        last_timestep = self.timesteps - 1
        if n_timesteps < self.timesteps:  # pad inputs and mask
            padded_input = np.zeros((n_samples, self.timesteps, n_inputdim))
            padded_input[:, :n_timesteps, :] = inputs[:, :, :]
            inputs = padded_input
            last_timestep = n_timesteps - 1
        elif n_timesteps > self.timesteps: # truncate inputs and mask, take the last self.timesteps
            inputs = inputs[:,n_timesteps - self.timesteps:, :]

        q = q_values.eval(session=session, feed_dict={q_inputs: [inputs]})
        best_actions = np.amax(q, axis=2)
        return best_actions[:, last_timestep]


# implemented using tflearn Trainer class.
# Packed everything a single computation graph, so we have a single train op to optimize.
# Previously, we had to first compute the target Q value separately and pass it into the graph.
# Now all data can be passed in together.
def build_drqn_tflearn(inputs, n_hidden, n_actions, reuse=False):
    """
    :param n_timesteps:
    :param n_inputdim:
    :param n_hidden:
    :param n_actions equivalent to number of actions
    :return:
    """
    net, hidden_states_1 = tflearn.lstm(inputs, n_hidden, return_seq=True, return_state=True, scope="lstm_1", reuse=reuse)
    q_values = tflearn.lstm(net, n_actions, return_seq=True, activation='linear', scope="lstm_2",reuse=reuse)
    q_values = tf.stack(q_values, axis=1) # tensor shape(None, n_timesteps, n_actions)
    return q_values


def build_tf_graph_drqn_tflearn(n_timesteps, n_inputdim, n_hidden, n_actions):
    tf.reset_default_graph()
    # Create shared deep q network
    q_inputs = tf.placeholder(tf.float32, [None, n_timesteps, n_inputdim])
    target_inputs = tf.placeholder(tf.float32, [None, n_timesteps, n_inputdim])
    # with tf.variable_scope("q_rnn") as scope:
    q_net = build_drqn_tflearn(q_inputs, n_hidden, n_actions, reuse=False)
    q_values = q_net
    target_net = build_drqn_tflearn(target_inputs, n_hidden, n_actions,reuse=True)
    target_values = target_net

    # Define cost and gradient update op
    a = tf.placeholder("float", [None, n_timesteps, n_actions])
    r = tf.placeholder("float", [None, n_timesteps]) # reward placeholder
    # gamma = tf.placeholder("float", [1])  # gamma placeholder
    gamma = tf.constant(0.99)
    action_q_values = tf.reduce_sum(tf.multiply(q_values, a), reduction_indices=2)  # shape [None, n_timesteps]
    # compute td cost as mean square error of target q and predicted q
    y = r + gamma * tf.reduce_max(target_values, reduction_indices=2)
    cost = tflearn.mean_square(action_q_values, y)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)


    # Define a train op
    train_op = tflearn.TrainOp(loss=cost, optimizer=optimizer,
                              # metric=tflearn.metrics.R2(),
                               batch_size=64)

    graph_ops = {"q_inputs": q_inputs,
                "q_values": q_values,
                 "target_inputs": target_inputs,
                 "target_values": target_values,
                "a": a,
                "r": r,
                # "gamma": gamma,
                "train_op":train_op}
    return graph_ops


def train_tflearn(graph_ops, train_buffer, val_buffer,n_epoch=16,
                  tensorboard_dir="/temp/tflearn/", run_id="test_run", load_checkpoint=False, load_ckpt_path="", save_ckpt_path=""):
    """
    Treat our offline data as the experience replay buffer and we only train on "experience"
    Data could be provided with the experience buffer (list of (s,a,r,s') tuples)
    1. experience buffer could go through data in order, in chunks of 64
    2. randomly sample a batch of 64 samples
    """
    # unpack graph_ops
    q_inputs = graph_ops["q_inputs"]
    target_inputs = graph_ops["target_inputs"]
    a = graph_ops["a"]
    r = graph_ops["r"]
    train_op = graph_ops["train_op"]


    trainer = tflearn.Trainer(train_ops=train_op,
                                  tensorboard_dir=tensorboard_dir,
                                  checkpoint_path=save_ckpt_path,
                                   max_checkpoints=3,
                                  tensorboard_verbose=2)


    if load_checkpoint:
        checkpoint = tf.train.latest_checkpoint(load_ckpt_path)  # can be none of no checkpoint exists
        print("Checkpoint filename: " + checkpoint)
        if checkpoint:
            trainer.restore(checkpoint, verbose=True)
            print('Checkpoint loaded.')

    train_batch = train_buffer.sample_in_order(train_buffer.buffer_sz)

    # make sure that batches are over multiple timesteps, should be of shape (batch_sz, n_timesteps, ?)
    s_batch_train = stack_batch(train_batch[:, :, 0])  # current states
    a_batch_train = stack_batch(train_batch[:, :, 1])  # actions
    r_batch_train = stack_batch(train_batch[:, :, 2])  # rewards
    sp_batch_train = stack_batch(train_batch[:, :, 3])  # next states

    val_batch = val_buffer.sample_in_order(val_buffer.buffer_sz)

    # make sure that batches are over multiple timesteps, should be of shape (batch_sz, n_timesteps, ?)
    s_batch_val = stack_batch(val_batch[:, :, 0])  # current states
    a_batch_val = stack_batch(val_batch[:, :, 1])  # actions
    r_batch_val = stack_batch(val_batch[:, :, 2])  # rewards
    sp_batch_val = stack_batch(val_batch[:, :, 3])  # next states

    # Training for 10 epochs.
    trainer.fit({q_inputs: s_batch_train, a: a_batch_train, r: r_batch_train, target_inputs: sp_batch_train},
                val_feed_dicts={q_inputs: s_batch_val, a: a_batch_val, r: r_batch_val, target_inputs: sp_batch_val},
                n_epoch=n_epoch, snapshot_epoch=True,run_id=run_id)


def stack_batch(batch):
    stacked_batch = np.array([[np.array(batch[i, j]) for j in xrange(batch.shape[1])] for i in xrange(batch.shape[0])])
    return stacked_batch




# Below: Deprecated code that doesn't use tflearn trainer.

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
    q_values = tflearn.lstm(net, n_actions, return_seq=True, activation='linear', name="lstm_2")
    q_values = tf.stack(q_values, axis=1) # tensor shape(None, n_timesteps, n_actions)
    return inputs, q_values


def build_tf_graph_drqn(n_timesteps, n_inputdim, n_hidden, n_actions):
    tf.reset_default_graph()
    # Create shared deep q network
    q_inputs, q_net = build_drqn(n_timesteps, n_inputdim, n_hidden, n_actions)
    net_params = tf.trainable_variables()
    q_values = q_net

    # Define cost and gradient update op
    a = tf.placeholder("float", [None, n_timesteps, n_actions])
    y = tf.placeholder("float", [None, n_timesteps])
    action_q_values = tf.reduce_sum(tf.multiply(q_values, a), reduction_indices=2)  # shape [None, n_timesteps]
    # compute td cost as mean square error of target q and predicted q
    cost = tflearn.mean_square(action_q_values, y)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    grad_update = optimizer.minimize(cost, var_list=net_params)

    # TODO: for now we will use a single Q network. Since we use offline data, there is less of a risk to instability.

    graph_ops = {"q_inputs": q_inputs,
                 "q_values": q_values,
                 "a": a,
                 "y": y,
                 "grad_update": grad_update}
    return graph_ops


def train(drqn_model, session, saver, experience_buffer, gamma=0.99, batch_sz=16, n_epoch=16,
          load_checkpoint=False, checkpoint_interval=1000, ckpt_path=""):
    """
    Treat our offline data as the experience replay buffer and we only train on "experience"
    Data could be provided with the experience buffer (list of (s,a,r,s') tuples)
    1. experience buffer could go through data in order, in chunks of 64
    2. randomly sample a batch of 64 samples
    """


    summary_ops = build_summaries(tf.summary.scalar, tf.summary.merge_all)
    summary_op = summary_ops[-1]

    graph_ops = drqn_model.graph_ops
    # unpack graph_ops
    q_inputs = graph_ops["q_inputs"]
    q_values = graph_ops["q_values"]
    a = graph_ops["a"]
    y = graph_ops["y"]
    grad_update = graph_ops["grad_update"]

    summary_placeholders, assign_ops, summary_op = summary_ops

    if load_checkpoint == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        saver.restore(session, ckpt_path)

    training_steps = int(n_epoch * (experience_buffer.buffer_sz / batch_sz))
    # one training step corresponds to one update to the Q network

    for t in xrange(training_steps):
        print("Training step: {}".format(t))
        # traces is a list/batch of experience traces. Each trace is a tuple of state_action_Data and rewards.
        train_batch = experience_buffer.sample_in_order(batch_sz)

        # make sure that batches are over multiple timesteps, should be of shape (batch_sz, n_timesteps, ?)
        s_batch = stack_batch(train_batch[:, :, 0])  # current states
        a_batch = stack_batch(train_batch[:, :, 1])  # actions
        r_batch = stack_batch(train_batch[:, :, 2])  # rewards
        sp_batch = stack_batch(train_batch[:, :, 3])  # next states

        y_batch = r_batch + gamma * np.amax(q_values.eval(session=session, feed_dict={q_inputs: sp_batch}), axis=2)

        # Update the network with our target values
        session.run(grad_update, feed_dict={y: y_batch,
                                            a: a_batch,
                                            q_inputs: s_batch})

        # Save model progress
        if t % checkpoint_interval == 0:
            saver.save(session, ckpt_path, global_step=t)


def evaluate(session, graph_ops, saver, ckpt_path, n_timesteps, eval_buffer):
    saver.restore(session, ckpt_path)
    print("Restored model weights from ", ckpt_path)

    q_inputs = graph_ops["q_inputs"]
    q_values = graph_ops["q_values"]

    n_eval_episodes = eval_buffer.buffer_sz
    for i in xrange(n_eval_episodes):
        # get initial state
        ep_reward = 0.0
        terminal = False

        train_batch = eval_buffer.sample_in_order(batch_sz=1)

        # make sure that batches are over multiple timesteps, should be of shape (batch_sz, n_timesteps, ?)
        s_batch = stack_batch(train_batch[:, :, 0])  # current states
        a_batch = stack_batch(train_batch[:, :, 1])  # actions
        r_batch = stack_batch(train_batch[:, :, 2])  # rewards
        sp_batch = stack_batch(train_batch[:, :, 3])  # next states

        Q = q_values.eval(session=session, feed_dict={q_inputs: s_batch})


def build_summaries(scalar_summary, merge_all_summaries):
    episode_reward = tf.Variable(0.)
    scalar_summary("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    scalar_summary("Qmax Value", episode_ave_max_q)
    logged_epsilon = tf.Variable(0.)
    scalar_summary("Epsilon", logged_epsilon)
    # Threads shouldn't modify the main graph, so we use placeholders
    # to assign the value of every summary (instead of using assign method
    # in every thread, that would keep creating new ops in the graph)
    summary_vars = [episode_reward, episode_ave_max_q, logged_epsilon]
    summary_placeholders = [tf.placeholder("float")
                            for i in range(len(summary_vars))]
    assign_ops = [summary_vars[i].assign(summary_placeholders[i])
                  for i in range(len(summary_vars))]
    summary_op = merge_all_summaries()
    return summary_placeholders, assign_ops, summary_op
