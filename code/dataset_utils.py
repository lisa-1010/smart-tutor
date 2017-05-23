# data_generator.py
# @author: Lisa Wang
# @created: Apr 2017
#
#===============================================================================
# DESCRIPTION:
# To handle data loading, and offline dataset creation to train dynamics
# models.
#===============================================================================
# CURRENT STATUS: In progress, working
#===============================================================================
# USAGE: from data_generator import *


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pickle
import numpy as np
import tensorflow as tf

# Custom Modules
from filepaths import *
from constants import *


def load_data(filename=None):
    data = pickle.load(open(filename, 'rb+'))
    return data


def preprocess_data_for_dqn(data, reward_model="sparse"):
    """
    Creates n_students traces of (s,a,r,s') tuples which can be loaded into the experience replay buffer.
    Each student yields one trace
    :param data:
    :param reward_model: "sparse" or "dense".
    If "sparse", then reward = percentage of skills learned at last timestep, 0 everywhere else
    If "dense", reward = percentage of skills learned at every timestep
    :return:
    """
    n_students = len(data)
    n_timesteps = len(data[0])
    exer = data[0][0][0]
    n_concepts = len(exer)

    all_traces = []
    for i in xrange(n_students):
        trace = []
        for t in xrange(n_timesteps - 1):
            cur_sample = data[i][t]
            next_sample = data[i][t + 1]
            exer, perf, knowl = cur_sample
            next_exer, next_perf, next_knowl = next_sample
            next_exer_ix = np.argmax(next_exer)

            s = np.zeros(2 * len(exer))
            if perf == 1:
                s[:len(exer)] = exer
            else:
                s[len(exer):] = exer

            a = np.array(next_exer)
            r = 0.0
            if reward_model == "dense" or (reward_model == "sparse" and t == n_timesteps - 2):
                # t = n_timesteps - 2 is last timestep we are considering, since next_knowl is from t+1
                r = np.mean(next_knowl)

            sp = np.zeros(2 * len(next_exer))
            if next_perf == 1:
                sp[:len(next_exer)] = next_exer
            else:
                sp[len(exer):] = next_exer

            trace.append([s,a,r,sp])
        all_traces.append(trace)
    return all_traces


def preprocess_data_for_rnn(data):
    n_students = len(data)
    n_timesteps = len(data[0])
    exer = data[0][0][0]
    n_concepts = len(exer)
    n_inputdim = 2 * n_concepts
    n_exercises = n_concepts
    n_outputdim = n_exercises

    input_data = np.zeros((n_students, n_timesteps, n_inputdim))
    output_mask = np.zeros((n_students, n_timesteps, n_outputdim))
    target_data = np.zeros((n_students, n_timesteps, n_outputdim))
    for i in xrange(n_students):
        for t in xrange(n_timesteps-1):
            cur_sample = data[i][t]
            next_sample = data[i][t+1]
            exer, perf, knowl = cur_sample
            next_exer, next_perf, next_knowl = next_sample
            next_exer_ix = np.argmax(next_exer)

            observ = np.zeros(2*len(exer))
            if perf == 1:
                observ[:len(exer)] = exer
            else:
                observ[len(exer):] = exer

            input_data[i,t,:] = observ[:]

            output_mask[i,t,next_exer_ix] = 1
            target_data[i,t,next_exer_ix] = next_perf

    return input_data, output_mask, target_data


def save_rnn_data(input_data, output_mask, target_data, filename):
    np.save("{}{}_input_data.npy".format(RNN_DATA_DIR, filename), input_data)
    np.save("{}{}_output_mask.npy".format(RNN_DATA_DIR, filename), output_mask)
    np.save("{}{}_target_data.npy".format(RNN_DATA_DIR, filename), target_data)


def load_rnn_data(filename):
    input_data = np.load("{}{}_input_data.npy".format(RNN_DATA_DIR, filename))
    output_mask = np.load("{}{}_output_mask.npy".format(RNN_DATA_DIR, filename))
    target_data = np.load("{}{}_target_data.npy".format(RNN_DATA_DIR, filename))
    return input_data, output_mask, target_data


def convert_to_rnn_input(action, observation):
    concept_vec = action.conceptvec
    input = np.zeros(2  * len(concept_vec))
    if observation == 1: # if student solved
        input[:len(concept_vec)] = concept_vec
    else:
        input[len(concept_vec):] = concept_vec
    return input

