# data_generator.py
# @author: Lisa Wang
# @created: Apr 2017
#
#===============================================================================
# DESCRIPTION:
# To handle data loading, and offline dataset creation to train dynamics
# models.
# Vaguely following Deepmind's Sonnet examples.
# https://github.com/deepmind/sonnet/blob/master/sonnet/examples/dataset_shakespeare.py
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

def load_data(filename=None):
    data = pickle.load(open(filename, 'rb+'))
    return data


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