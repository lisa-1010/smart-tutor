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

def dkt_forwardsearch_single_recurse(n_concepts, dkt, sim, horizon, history_len):
    '''
    Given the current history, compute the q-values of the optimal policy according to the dkt
    under both SEMISPARSE and SPARSE rewards. Also, a list of optimal actions and q-values
    according to SEMISPARSE and SPARSE obtained from the optimal policy of the dkt under the true environment
    sim is also returned.
    :param n_concepts: number of concepts
    :param dkt: an RnnStudentSim-like object
    :param sim: an RnnStudentSim-like object
    :param horizon: the horizon
    :param history_len: length of current history
    Return (
        learned semisparse value,
        learned sparse value,
        sim semisparse value of learned opt policy,
        sim sparse value of learned opt policy,
        list of actions and learned semisparse q-values along sim trajectory,
        list of actions and learned sparse q-values along sim trajectory,
        list of actions and sim semisparse q-values along sim trajectory,
        list of actions and sim sparse q-values along sim trajectory,
    '''
    #six.print_('history len {}'.format(history_len))
    
    assert history_len <= horizon
    if history_len == horizon:
        # we have now finished running horizon number of actions, it's time for the final reward
        dkt_probs = sanitize_probs(n_concepts, dkt.sample_observations())
        sim_probs = sanitize_probs(n_concepts, sim.sample_observations())
        
        semisparse_reward = np.mean(dkt_probs)
        sparse_reward = np.prod(dkt_probs)
        sim_semisparse_reward = np.mean(sim_probs)
        sim_sparse_reward = np.prod(sim_probs)
        
        #six.print_('sim probs {}'.format(sim_probs))
        
        return (
            semisparse_reward,
            sparse_reward,
            sim_semisparse_reward,
            sim_sparse_reward,
            [],
            [],
            [],
            []
        )
    
    
    # go over all possible next actions and observations to compute the qvalues
    # then use qvalues to find best action and thus best value
    # accumulate best action in the lists as well
    ssqvalues = np.zeros((n_concepts,))
    sqvalues = np.zeros((n_concepts,))
    sim_ssqvalues = np.zeros((n_concepts,))
    sim_sqvalues = np.zeros((n_concepts,))
    
    next_probs = sanitize_probs(n_concepts, dkt.sample_observations())
    sim_next_probs = sanitize_probs(n_concepts, sim.sample_observations())
    
    next_ss_list = [[] for _ in six.moves.range(n_concepts)]
    next_s_list = [[] for _ in six.moves.range(n_concepts)]
    next_sim_ss_list = [[] for _ in six.moves.range(n_concepts)]
    next_sim_s_list = [[] for _ in six.moves.range(n_concepts)]
    
    if False:
        six.print_('history len {}'.format(history_len))
        six.print_('next probs {}'.format(next_probs))
        six.print_('sim next probs {}'.format(sim_next_probs))
    
    for next_action in six.moves.range(n_concepts):
        curr_ssq = 0.0
        curr_sq = 0.0
        curr_sim_ssq = 0.0
        curr_sim_sq = 0.0
        
        for next_ob in (0,1):
            # advance the state
            next_dkt = dkt.copy()
            next_dkt.advance_simulator(st.make_student_action(n_concepts,next_action), next_ob)
            next_sim = sim.copy()
            next_sim.advance_simulator(st.make_student_action(n_concepts,next_action), next_ob)
            
            next_ssv,next_sv,next_sim_ssv,next_sim_sv,ss_list,s_list,sim_ss_list,sim_s_list = dkt_forwardsearch_single_recurse(
                n_concepts, next_dkt, next_sim, horizon, history_len+1)
            
            next_ss_list[next_action].append(ss_list)
            next_s_list[next_action].append(s_list)
            next_sim_ss_list[next_action].append(sim_ss_list)
            next_sim_s_list[next_action].append(sim_s_list)
            
            next_p = (1 - next_ob) * (1.0 - next_probs[next_action]) + next_ob * (next_probs[next_action])
            next_sim_p = (1 - next_ob) * (1.0 - sim_next_probs[next_action]) + next_ob * (sim_next_probs[next_action])
            
            curr_ssq += next_ssv * next_p
            curr_sq += next_sv * next_p
            curr_sim_ssq += next_sim_ssv * next_sim_p
            curr_sim_sq += next_sim_sv * next_sim_p
            
            if False:
                six.print_('action {} ob {}'.format(next_action, next_ob))
                six.print_('    ssv {} sv {} sim ssv {} sim sv {}'.format(next_ssv, next_sv, next_sim_ssv, next_sim_sv))
                six.print_('    next p {} next sim p {}'.format(next_p, next_sim_p))
                six.print_('    ssq {} sq {} sim ssq {} sim sq {}'.format(curr_ssq, curr_sq, curr_sim_ssq, curr_sim_sq))
        
        ssqvalues[next_action] = curr_ssq
        sqvalues[next_action] = curr_sq
        sim_ssqvalues[next_action] = curr_sim_ssq
        sim_sqvalues[next_action] = curr_sim_sq
       
    # find best actions
    ss_optimal_action = np.argmax(ssqvalues)
    s_optimal_action = np.argmax(sqvalues)
    
    # find corresponding observations from sim
    ss_sim_ob = int(sim_next_probs[ss_optimal_action])
    s_sim_ob = int(sim_next_probs[s_optimal_action])
    
    # update the relevant lists
    next_ss_list[ss_optimal_action][ss_sim_ob].append((ss_optimal_action,ssqvalues))
    next_s_list[s_optimal_action][s_sim_ob].append((s_optimal_action,sqvalues))
    next_sim_ss_list[ss_optimal_action][ss_sim_ob].append((ss_optimal_action,sim_ssqvalues))
    next_sim_s_list[s_optimal_action][s_sim_ob].append((s_optimal_action,sim_sqvalues))
    
    if False:
        six.print_('ss opt act {} s opt act {}'.format(ss_optimal_action, s_optimal_action))
        six.print_('ss sim ob {} s sim ob {}'.format(ss_sim_ob, s_sim_ob))
    
    # return list
    return (
        ssqvalues[ss_optimal_action],
        sqvalues[s_optimal_action],
        sim_ssqvalues[ss_optimal_action],
        sim_sqvalues[s_optimal_action],
        next_ss_list[ss_optimal_action][ss_sim_ob],
        next_s_list[s_optimal_action][s_sim_ob],
        next_sim_ss_list[ss_optimal_action][ss_sim_ob],
        next_sim_s_list[s_optimal_action][s_sim_ob]
    )
    

def dkt_forwardsearch_single(n_concepts, model_id, checkpoints, horizon, use_mem):
    '''
    Use forward search to find value of the optimal policy of dkt executed in sim and other information.
    '''
    if not use_mem:
        model_list = []
        for chkpt in checkpoints:
            model = dmc.DynamicsModel(model_id, timesteps=horizon, load_checkpoint=False)
            model.load(chkpt)
            model_list.append(model)
        dkt = dmc.RnnStudentSimEnsemble(model_list)
    else:
        mem_array_list = []
        for chkpt in checkpoints:
            mem_arrays = np.load(chkpt)['mem_arrays']
            mem_array_list.append(mem_arrays)
        dkt = dmc.RnnStudentSimMemEnsemble(n_concepts, mem_array_list)
    
    concept_tree = cdg.ConceptDependencyGraph()
    concept_tree.init_default_tree(n_concepts)
    sim = st.RnnStudent2SimExact(concept_tree)
    
    if False:
        six.print_('Semisparse Value {}'.format(next_ssv))
        six.print_('Spares Value {}'.format(next_sv))
        six.print_('Semisparse Value Sim {}'.format(next_sim_ssv))
        six.print_('Sparse Value Sim {}'.format(next_sim_sv))
        six.print_('Semisparse Q-Values along sim trajectory {}'.format(ss_list))
        six.print_('Sparse Q-Values along sim trajectory {}'.format(s_list))
        six.print_('Semisparse Sim Q-Values along sim trajectory {}'.format(sim_ss_list))
        six.print_('Sparse Sim Q-Values along sim trajectory {}'.format(sim_s_list))
    
    return dkt_forwardsearch_single_recurse(n_concepts, dkt, sim, horizon, 0)


def dkt_forwardsearch_chunk(params, horizon, runstartix, chunk_num_runs, use_mem):
    fsdata = []
    
    for offset in six.moves.range(chunk_num_runs):
        fsdata.append([])
        
        r = runstartix + offset
        
        for ep in params.saved_epochs:
            print('=====================================')
            print('---------- Rep {:2d} Epoch {:2d} ----------'.format(r, ep))
            print('=====================================')
            
            if not use_mem:
                # compute checkpoint name
                checkpoint_name = params.checkpoint_pat.format(params.run_name, r, ep)
                checkpoint_path = '{}/{}'.format(params.dir_name,checkpoint_name)
            else:
                # compute outfile name
                mem_name = params.mem_pat.format(params.run_name, r, ep)
                checkpoint_path = '{}/{}'.format(params.dir_name,mem_name)
            
            # forward search
            rundata = dkt_forwardsearch_single(params.n_concepts, params.model_id, [checkpoint_path], horizon, use_mem)
            
            fsdata[-1].append(rundata)
            
            six.print_('Finished.')
    
    return fsdata

def dkt_forwardsearch(params, horizon, use_mem=False):
    '''
    Runs forward search to extract their optimal policies and its performance in the simulator
    '''
    n_jobs = min(5, params.num_runs) # seems like there are problems on windows with multiple threads
    # need to be a multiple of number of jobs so I don't have to deal with uneven leftovers
    assert(params.num_runs % n_jobs == 0)
    runs_per_job = int(params.num_runs / n_jobs)
    
    fsdata = []
    
    returned_data = list(
        Parallel(n_jobs=n_jobs)(delayed(dkt_forwardsearch_chunk)(params,horizon,startix,runs_per_job,use_mem)
                                for startix in six.moves.range(0,params.num_runs,runs_per_job)))
    
    for d in returned_data:
        fsdata.extend(d)
    
    statfile = 'fsearch-{}-horizon{}.pickle'.format(params.run_name, horizon)
    statpath = '{}/{}'.format(params.dir_name,statfile)
    with open(statpath, 'wb') as f:
        pickle.dump(fsdata,f)
    
def dkt_forwardsearch_ensemble(trainparams,testparams):
    '''
    Given a set of runs, test ensemble models with forward search
    '''
    
    fsdata = []
    
    for en in six.moves.range(testparams.ensemble_split):
        # compute how many runs to use
        curr_num_runs = int((en+1) * trainparams.num_runs / testparams.ensemble_split)
        fsdata.append([])
        for ep in trainparams.saved_epochs:
            print('=================================================')
            print('---------- Split {:1d}/{:1d} Runs {} Epoch {:2d} ----------'.format(en+1,testparams.ensemble_split, curr_num_runs, ep))
            print('=================================================')
            
            # create the checkpoints of all models
            curr_checkpoints = []
            for r in six.moves.range(curr_num_runs):
                if not testparams.use_mem:
                    checkpoint_name = trainparams.checkpoint_pat.format(trainparams.run_name, r, ep)
                    checkpoint_path = '{}/{}'.format(trainparams.dir_name,checkpoint_name)
                else:
                    mem_name = trainparams.mem_pat.format(trainparams.run_name, r, ep)
                    checkpoint_path = '{}/{}'.format(trainparams.dir_name,mem_name)
                curr_checkpoints.append(checkpoint_path)
            
            # do the forward search
            returned_data = dkt_forwardsearch_single(
                trainparams.n_concepts, trainparams.model_id, curr_checkpoints, testparams.horizon, testparams.use_mem)
            
            # update stats
            fsdata[-1].append(returned_data)
            
    
    # save stats
    stat_name = 'fsearchensemble-{}-horizon{}.pickle'.format(trainparams.run_name, testparams.horizon)
    stats_path = '{}/{}'.format(trainparams.dir_name,stat_name)
    with open(stats_path, 'wb') as f:
        pickle.dump(fsdata,f)