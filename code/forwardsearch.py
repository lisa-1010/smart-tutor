##############################################################################
# Implements simple monte carlo forward search to compute
# q-values and optimal policy of models. Assume deterministic environment.
##############################################################################

import numpy as np
import scipy as sp
import tensorflow as tf
import tflearn

import time
import copy
import pickle
import multiprocessing as mp
import six
import os
import random
import itertools

import constants
import data_generator as dg
import concept_dependency_graph as cdg
import student as st
import dynamics_model_class as dmc
import dataset_utils

from simple_mdp import SimpleMDP
from joblib import Parallel, delayed

from helpers import * # helper functions
from simple_mdp import create_custom_dependency

def dkt_forwardsearch_single_recurse(dktstate):
    '''
    Given a dktstate, compute the q-values for all next actions.
    Returns the list of q-values and optimal actions along the optimal path.
    The list is in reverse order i.e. the first element is the last step.
    Uses the lowest indexed action in the case of ties.
    TODO
    '''
    # debugging print
    if False:
        pass
    
    if dktstate.is_terminal:
        # we've finished
        # all actions are the same so just use action 0
        qvalues = np.full((dktstate.n_concepts,),dktstate.reward())
        optimal_action = 0
        return [(optimal_action,qvalues)]
    
    # go over all possible next actions to compute the qvalues
    qvalues = np.zeros((dktstate.n_concepts,))
    next_lists = []
    curr_reward = dktstate.reward()
    for next_action in six.moves.range(n_concepts):
        # get next list
        next_dktstate = dktstate.real_world_perform(st.make_student_action(next_action))
        next_list = dkt_forwardsearch_single_recurse(next_dktstate)
        next_lists.append(next_list)
        # extract next best value
        next_optimal_action,next_qvalues = next_list[-1]
        next_value = next_qvalues[next_optimal_action]
        qvalues[next_action] = curr_reward + next_value
    # find best action
    optimal_action = np.argmax(qvalues)[0]
    
    # return list
    next_lists[optimal_action].append((optimal_action,qvalues))
    

def dkt_forwardsearch_single(n_concepts, model_id, checkpoint, horizon, outfile):
    '''
    TODO
    '''
    # load up the model
    dmodel = dmc.DynamicsModel(model_id, timesteps=horizon, load_checkpoint=False)
    dmodel.load(checkpoint)
    # wrap
    dkt = dmc.RnnStudentSim(dmodel)
    
    # compute the number of branches i.e |num actions|*2
    index_base = n_concepts * 2
    
    # initialize all to zero
    mem_arrays = [None] * (horizon+1)
    for i in six.moves.range(horizon+1):
        mem_arrays[i] = np.zeros((num_histories(index_base,i),n_concepts))
    
    # start populating the mem arrays recursive
    dkt_memoize_single_recurse(n_concepts, dkt, horizon, 1, 0, mem_arrays)
    
    # finished so write it
    np.savez(outfile, mem_arrays=mem_arrays)