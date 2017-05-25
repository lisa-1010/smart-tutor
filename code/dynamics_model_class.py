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
import dataset_utils as d_utils
import models_dict_utils
FLAGS = tf.flags.FLAGS

# n_timesteps = 10
n_inputdim = 20
n_hidden = 32
n_outputdim = 10


class DynamicsModel(object):

    def __init__(self, model_id, timesteps=100, dropout=0.5, load_checkpoint=False):
        print('Loading RNN dynamics model...')

        # if timesteps:
        #     # if provided as an argument, overwrite n_timesteps from the model
        #     n_timesteps = timesteps
        tf.reset_default_graph()
        self.timesteps = timesteps
        self.model_dict = models_dict_utils.load_model_dict(model_id)
        self.net, self.hidden_1, self.hidden_2 = self._build_regression_lstm_net(n_timesteps=timesteps,
                                                                                 n_inputdim=self.model_dict["n_inputdim"],
                                                                                 n_hidden=self.model_dict["n_hidden"],
                                                                                 n_outputdim=self.model_dict["n_outputdim"],
                                                                                 dropout=dropout)

        tensorboard_dir = '../tensorboard_logs/' + model_id + '/'
        checkpoint_dir = '../checkpoints/' + model_id + '/'
        checkpoint_path = checkpoint_dir + '_/'
        print("Directory path for tensorboard summaries: {}".format(tensorboard_dir))
        print("Checkpoint directory path: {}".format(checkpoint_dir))

        utils.check_if_path_exists_or_create(tensorboard_dir)
        utils.check_if_path_exists_or_create(checkpoint_dir)

        self.model = tflearn.DNN(self.net, tensorboard_verbose=2, tensorboard_dir=tensorboard_dir, \
                                checkpoint_path=checkpoint_path, max_checkpoints=3)

        if load_checkpoint:
            checkpoint = tf.train.latest_checkpoint(checkpoint_dir)  # can be none of no checkpoint exists
            print ("Checkpoint filename: " + checkpoint)
            if checkpoint:
                self.model.load(checkpoint, weights_only=True, verbose=True)
                print('Checkpoint loaded.')
            else:
                print('No checkpoint found. ')

        print('Model loaded.')


    def _build_regression_lstm_net(self, n_timesteps=10, n_inputdim=n_inputdim, n_hidden=n_hidden,
                                           n_outputdim=n_outputdim, dropout=0.5):
        net = tflearn.input_data([None, n_timesteps, n_inputdim],dtype=tf.float32, name='input_data')
        output_mask = tflearn.input_data([None, n_timesteps, n_outputdim], dtype=tf.float32, name='output_mask')
        net, hidden_states_1 = tflearn.lstm(net, n_hidden, return_seq=True, return_state=True, dropout=dropout, name="lstm_1")
        net, hidden_states_2 = tflearn.lstm(net, n_outputdim, activation='sigmoid', return_seq=True, return_state=True, dropout=dropout, name="lstm_2")
        net = tf.stack(net, axis=1)
        net = net * output_mask
        net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                                 loss='mean_square')
        return net, hidden_states_1, hidden_states_2


    def train(self, train_data, n_epoch=64, load_checkpoint=True):
        """

        :param train_data: tuple (input_data, output_mask, output_data)
        :param n_epoch: number of epochs to train for
        :param load_checkpoint: whether to train from checkpoint or from scratch
        :return:
        """
        input_data, output_mask, output_data = train_data
        tf.reset_default_graph()
        date_time_string = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        run_id = "{}".format(date_time_string)
        self.model.fit([input_data, output_mask], output_data, n_epoch=n_epoch, validation_set=0.1, run_id=run_id)


    def predict(self, input_data):
        """
        :param input_data: of shape (n_samples, n_timesteps, n_inputdim).
        :return:
        """
        n_samples, n_timesteps, n_inputdim = input_data.shape
        assert(n_inputdim == self.model_dict["n_inputdim"]), "input dimension of data doesn't match the model."
        n_outputdim = self.model_dict["n_outputdim"]
        output_mask = np.ones((n_samples, n_timesteps, n_outputdim))
        if n_timesteps < self.timesteps:  # pad inputs and mask
            padded_input = np.zeros((n_samples, self.timesteps, n_inputdim))
            padded_input[:, :n_timesteps, :] = input_data[:, :, :]
            input_data = padded_input
            padded_mask = np.zeros((n_samples, self.timesteps, n_outputdim))
            padded_mask[:,:n_timesteps,:] = output_mask[:,:,:]
            output_mask = padded_mask
        elif n_timesteps > self.timesteps: # truncate inputs and mask
            input_data = input_data[:, :self.timesteps, :]
            output_mask = output_mask[:, :self.timesteps, :]
        tf.reset_default_graph()
        return self.model.predict([input_data, output_mask])


class RnnStudentSim(object):
    '''
    A model-based simulator for a student. Maintains its own internal hidden state.
    Currently model can be shared because only the history matters
    This is just a template.
    '''

    def __init__(self, model):
        self.model = model
        self.seq_max_len = model.timesteps
        self.sequence = [] # will store up to seq_max_len
        pass


    def sample_observations(self):
        """
        Returns list of probabilities
        """
        # special case when self.sequence is empty
        if not self.sequence:
            return None
        else:
            # turns the list of input vectors, into a numpy matrix of shape (1, n_timesteps, 2*n_concepts)
            # We need the first dimension since the network expects a batch.
            rnn_input_sequence = np.expand_dims(np.array(self.sequence), axis=0)
            pred = self.model.predict(rnn_input_sequence)
            t = len(self.sequence)

            prob_success_action = pred[0][t-1]
            # observation is a probability
            return prob_success_action

    def advance_simulator(self, action, observation):
        '''
        Given next action and observation, advance the internal hidden state of the simulator.
        '''
        input = d_utils.convert_to_rnn_input(action, observation)
        if len(self.sequence) == self.seq_max_len:
            self.sequence = self.sequence[1:] + [input]
        else:
            self.sequence.append(input)


    def copy(self):
        '''
        Make a copy of the current simulator.

        '''
        sim_copy = RnnStudentSim(self.model)
        sim_copy.sequence = self.sequence[:] # deep copy
        return sim_copy
