# dynamics_model_class.py
# @author: Lisa Wang
# @created: Apr 25 2017
#
#===============================================================================
# DESCRIPTION:
# exports dynamics model class, allows training and predicts next observation
#
#===============================================================================
# CURRENT STATUS: In progress
#===============================================================================
# USAGE: from dynamics_model_class import *

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import tensorflow as tf
import tflearn
import numpy as np

import utils

FLAGS = tf.flags.FLAGS

n_timesteps = 10
n_inputdim = 20
n_hidden = 32
n_outputdim = 10


class DynamicsModel(object):
    def __init__(self, model_id, timesteps=None, load_checkpoint=False):
        print('Loading RNN dynamics model...')

        # if timesteps:
        #     # if provided as an argument, overwrite n_timesteps from the model
        #     n_timesteps = timesteps
        tf.reset_default_graph()
        self.net = _build_regression_lstm_net(n_timesteps=n_timesteps, n_inputdim=n_inputdim, n_hidden=n_hidden,
                                         n_outputdim=n_outputdim)

        tensorboard_dir = '../tensorboard_logs/' + model_id + '/'
        checkpoint_path = '../checkpoints/' + model_id + '/'

        print("Directory path for tensorboard summaries: {}".format(tensorboard_dir))
        print("Checkpoint directory path: {}".format(checkpoint_path))

        utils.check_if_path_exists_or_create(tensorboard_dir)
        utils.check_if_path_exists_or_create(checkpoint_path)

        self.model = tflearn.DNN(net, tensorboard_verbose=2, tensorboard_dir=tensorboard_dir, \
                                checkpoint_path=checkpoint_path, max_checkpoints=3)

        if load_checkpoint:
            checkpoint = tf.train.latest_checkpoint(checkpoint_path)  # can be none of no checkpoint exists
            if checkpoint and os.path.isfile(checkpoint):
                self.model.load(checkpoint, weights_only=True, verbose=True)
                print('Checkpoint loaded.')
            else:
                print('No checkpoint found. ')

        print('Model loaded.')


    def _build_regression_lstm_net(self, n_timesteps=n_timesteps, n_inputdim=n_inputdim, n_hidden=n_hidden,
                                           n_outputdim=n_outputdim):
        net = tflearn.input_data([None, n_timesteps, n_inputdim],dtype=tf.float32, name='input_data')
        output_mask = tflearn.input_data([None, n_timesteps, n_outputdim], dtype=tf.float32, name='output_mask')
        net = tflearn.lstm(net, n_hidden, return_seq=True, name="lstm_1")
        net = tflearn.lstm(net, n_outputdim, return_seq=True, name="lstm_2")
        net = tf.stack(net, axis=1)
        preds = net
        net = net * output_mask
        net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                                 loss='mean_square')
        return net


    def train(self, train_data, load_checkpoint=True):
        input_data, output_mask, output_data = train_data
        tf.reset_default_graph()
        date_time_string = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        run_id = "{}".format(date_time_string)
        self.model.fit([input_data, output_mask], output_data, n_epoch=64, validation_set=0.1)


    def predict(self, input_data):
        n_samples, n_timesteps, n_inputdim = input_data.shape
        output_mask = np.ones((n_samples, n_timesteps, n_outputdim))
        tf.reset_default_graph()
        return self.model.predict([input_data, output_mask])


#
# Once RNN is tested, package it into a class.
# class RNNDynamicsModel(object):
#     def __init__(self, load_from_ckpt=True):
#         self.model = snt.VanillaRNN(128, )
#         pass
#
#
#     def train_model_with_offline_data(self, data):
#         pass
#
#
#     def predict_next_observation(self, history_observations):
#         pass
