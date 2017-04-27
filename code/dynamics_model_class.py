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

FLAGS = tf.flags.FLAGS

# n_timesteps = 10
n_inputdim = 20
n_hidden = 32
n_outputdim = 10


class DynamicsModel(object):

    def __init__(self, model_id, timesteps=10, load_checkpoint=False):
        print('Loading RNN dynamics model...')

        # if timesteps:
        #     # if provided as an argument, overwrite n_timesteps from the model
        #     n_timesteps = timesteps
        tf.reset_default_graph()
        self.net, self.hidden_1, self.hidden_2 = self._build_regression_lstm_net(n_timesteps=timesteps, n_inputdim=n_inputdim, n_hidden=n_hidden,
                                         n_outputdim=n_outputdim)

        tensorboard_dir = '../tensorboard_logs/' + model_id + '/'
        checkpoint_path = '../checkpoints/' + model_id + '/'

        print("Directory path for tensorboard summaries: {}".format(tensorboard_dir))
        print("Checkpoint directory path: {}".format(checkpoint_path))

        utils.check_if_path_exists_or_create(tensorboard_dir)
        utils.check_if_path_exists_or_create(checkpoint_path)

        self.model = tflearn.DNN(self.net, tensorboard_verbose=2, tensorboard_dir=tensorboard_dir, \
                                checkpoint_path=checkpoint_path, max_checkpoints=3)

        if load_checkpoint:
            checkpoint = tf.train.latest_checkpoint(checkpoint_path)  # can be none of no checkpoint exists
            if checkpoint and os.path.isfile(checkpoint):
                self.model.load(checkpoint, weights_only=True, verbose=True)
                print('Checkpoint loaded.')
            else:
                print('No checkpoint found. ')

        print('Model loaded.')


    def _build_regression_lstm_net(self, n_timesteps=10, n_inputdim=n_inputdim, n_hidden=n_hidden,
                                           n_outputdim=n_outputdim):
        net = tflearn.input_data([None, n_timesteps, n_inputdim],dtype=tf.float32, name='input_data')
        output_mask = tflearn.input_data([None, n_timesteps, n_outputdim], dtype=tf.float32, name='output_mask')
        net, hidden_states_1 = tflearn.lstm(net, n_hidden, return_seq=True, return_state=True, name="lstm_1")
        net, hidden_states_2 = tflearn.lstm(net, n_outputdim, return_seq=True, return_state=True, name="lstm_2")
        net = tf.stack(net, axis=1)
        net = tf.sigmoid(net) # to make sure that predictions are between 0 and 1.
        preds = net
        net = net * output_mask
        net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                                 loss='mean_square')
        return net, hidden_states_1, hidden_states_2


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


class RnnStudentSim(object):
    '''
    A model-based simulator for a student. Maintains its own internal hidden state.
    This is just a template.
    '''

    def __init__(self, data, model_id=None, seq_max_len=10):
        # what is data here?
        # Can I add more arguments?
        if model_id:
            self.model = DynamicsModel(model_id=model_id, timesteps=1, load_checkpoint=False)
        else:
            self.model = None
        # set initial_state to random?
        self.seq_max_len = seq_max_len
        self.sequence = [] # will store up to seq_max_len
        pass


    def sample_observation(self, action):
        """
        Samples a new observation given an action.
        :param action: of class StudentAction
        :return:
        """
        # turns the list of input vectors, into a numpy matrix of shape (1, n_timesteps, 2*n_concepts)
        # We need the first dimension since the network expects a batch.
        rnn_input_sequence = np.expand_dims(np.array(self.sequence), axis=0)
        pred = self.model.predict(rnn_input_sequence)

        prob_success_action = pred[:, -1, action.concept] # action.concept is an index
        # index into new_pred using action and return single probability
        # observation is a probability
        # return new_observ


    def sample_reward(self, action):
        """
        Samples a new reward given an action.
        :param action: of class StudentAction
        :return:
        """
        pass

    def advance_simulator(self, action, observation):
        '''
        Given next action and observation, advance the internal hidden state of the simulator.
        Question: Is action here a StudentAction object?
        '''
        input = d_utils.convert_to_rnn_input(action, observation)
        if len(self.sequence == self.seq_max_len):
            self.sequence = self.sequence[1:] + [input]
        else:
            self.sequence.append(input)


    def copy(self):
        '''
        Make a copy of the current simulator.

        '''
        sim_copy = RnnStudentSim()
        sim_copy.model = self.model
        sim_copy.sequence = self.sequence[:] # deep copy
        return sim_copy
