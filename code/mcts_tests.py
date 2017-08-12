#===============================================================================
# DESCRIPTION:
# Tests on DKT + MCTS
#===============================================================================

#===============================================================================
# CURRENT STATUS:
#===============================================================================

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
import exercise as exer
import dynamics_model_class as dmc
import dataset_utils

from simple_mdp import SimpleMDP
from joblib import Parallel, delayed

from mctslib.graph import *
from mctslib.mcts import *

from helpers import * # helper functions
from simple_mdp import create_custom_dependency

import mctslib.tree_policies as tree_policies
import mctslib.default_policies as default_policies
import mctslib.backups as backups


from mcts import \
    MCTS, \
    ExactGreedyPolicy, \
    DKTGreedyPolicy, \
    ActionNode, \
    StateNode, \
    DKTState, \
    StudentExactState, \
    DENSE,SEMISPARSE,SPARSE



def debug_visiter(node, data):
    print('Curr node id: {} n: {} q: {}'.format(id(node), node.n, node.q))
    print('Parent id: {}'.format(str(id(node.parent)) if node.parent is not None else 'None'))
    if isinstance(node, ActionNode):
        print('Action {}'.format(node.action.concept))
    elif isinstance(node, StateNode):
        print('State {} step: {}/{} r: {}'.format(node.state.model.student.knowledge, node.state.step, node.state.horizon, node.reward))
    else:
        print('Not action nor state')



def test_student_exact_single(dgraph, stud, horizon, n_rollouts, r_type):
    '''
    Performs a single trajectory with MCTS and returns the final true student knowlegde.
    '''
    n_concepts = dgraph.n

    # create the model and simulators
    student = stud.copy()
    student.reset()
    student.knowledge[0] = 1 # initialize the first concept to be known
    sim = st.StudentExactSim(student, dgraph)
    model = sim.copy()

    #rollout_policy = default_policies.immediate_reward
    rollout_policy = default_policies.RandomKStepRollOut(horizon+1)
    uct = MCTS(tree_policies.UCB1(1.41), rollout_policy,
               backups.monte_carlo)

    root = StateNode(None, StudentExactState(model, sim, 1, horizon, r_type)) # create root node of the tree, 1 is the step.
    for i in range(horizon):
        #print('Step {}'.format(i))
        best_action = uct(root, n=n_rollouts) # state action object
        #print('Current state: {}'.format(str(root.state)))
        #print(best_action.concept)

        # debug check for whether action is optimal

        if False:
            opt_acts = compute_optimal_actions(sim.dgraph, sim.get_knowledge())
            is_opt = best_action.concept in opt_acts  # check if predicted action is optimal

            if not is_opt:
                print('ERROR {} executed non-optimal action {}'.format(sim.get_knowledge(), best_action.concept))
                # now let's print out even more debugging information
                #breadth_first_search(root, fnc=debug_visiter)
                #return None

        # act in the real environment
        new_root = root.children[best_action].sample_state(real_world=True) # children of root action nodes, real_world=true advances simulator
        new_root.parent = None # cutoff the rest of the tree
        root = new_root
        #print('Next state: {}'.format(str(new_root.state)))
    return sim.get_knowledge()

def test_student_exact_chunk(n_trajectories, dgraph, student, horizon, n_rollouts, r_type):
    '''
    Runs a bunch of trajectories and returns the avg posttest score.
    For parallelization to run in a separate thread/process.
    '''
    acc = 0.0
    for i in xrange(n_trajectories):
        print('traj i {}'.format(i))
        k = test_student_exact_single(dgraph, student, horizon, n_rollouts, r_type)
        acc += np.mean(k)
    return acc


def test_student_exact():
    '''
    MCTS is now working.
    The number of rollouts required to be optimal grows very fast as a function of the horizon.
    Still, even if not fully optimal, MCTS is an extremely good approximation.

    Default student with horizon 10 needs about 50 rollouts is good
    learn prob 0.15 student with horizon 40 needs about 150 rollouts is good; gets about 0.94 which is 0.02 off from 0.96
    '''
    import concept_dependency_graph as cdg
    from simple_mdp import create_custom_dependency
    r_type = DENSE
    n_concepts = 4
    learn_prob = 0.5
    horizon = 6
    n_rollouts = 50
    n_trajectories = 100
    n_jobs = 8
    traj_per_job =  n_trajectories // n_jobs

    #dgraph = create_custom_dependency()

    dgraph = cdg.ConceptDependencyGraph()
    dgraph.init_default_tree(n_concepts)

    #student = st.Student(n=n_concepts,p_trans_satisfied=learn_prob, p_trans_not_satisfied=0.0, p_get_ex_correct_if_concepts_learned=1.0)
    student2 = st.Student2(n_concepts, transition_after)
    test_student = student2

    accs = Parallel(n_jobs=n_jobs)(delayed(test_student_exact_chunk)(traj_per_job, dgraph, test_student, horizon, n_rollouts, sparse_r) for _ in range(n_jobs))
    avg = sum(accs) / (n_jobs * traj_per_job)

    test_data = dg.generate_data(dgraph, student=test_student, n_students=1000, seqlen=horizon, policy='expert', filename=None, verbose=False)
    print('Number of jobs {}'.format(n_jobs))
    print('Trajectory per job {}'.format(traj_per_job))
    print('Average posttest true: {}'.format(expected_reward(test_data)))
    print('Average posttest mcts: {}'.format(avg))


def test_dkt_single(dgraph, sim, horizon, n_rollouts, model, r_type, dktcache, use_real):
    '''
    Performs a single trajectory with MCTS and returns the final true student knowledge.
    :param dktcache: a dictionary to use for the dkt cache
    '''
    n_concepts = dgraph.n

    # make the model
    model = dmc.RnnStudentSim(model)

    #rollout_policy = default_policies.immediate_reward
    rollout_policy = default_policies.RandomKStepRollOut(horizon+1)
    uct = MCTS(tree_policies.UCB1(1.41), rollout_policy,
               backups.monte_carlo) # 1.41 is sqrt (2), backups is from mcts.py

    root = StateNode(None, DKTState(model, sim, 1, horizon, r_type, dktcache, use_real))
    for i in range(horizon):
        #print('Step {}'.format(i))
        best_action = uct(root, n=n_rollouts)
        #print('Current state: {}'.format(str(root.state)))
        #print(best_action.concept)
        
        # return the largest q-value at the end
        if i == horizon-1:
            #print('Root q value: {}'.format(root.q))
            child = root.children[best_action]
            #print('--- Best Child action {}, q value {}'.format(child.action.concept, child.q))
            best_q_value = child.q

        # act in the real environment
        new_root = root.children[best_action].sample_state(real_world=True)
        new_root.parent = None # cutoff the rest of the tree
        root = new_root
        #print('Next state: {}'.format(str(new_root.state)))
    return sim.get_knowledge(), best_q_value

def test_dkt_chunk(n_trajectories, dgraph, s, model_id, chkpt, horizon, n_rollouts, r_type, dktcache=None, use_real=True):
    '''
    Runs a bunch of trajectories and returns the avg posttest score.
    For parallelization to run in a separate thread/process.
    '''
    # load the model
    # add 2 to the horizon since MCTS might look at horizon+1 steps
    if chkpt is not None:
        model = dmc.DynamicsModel(model_id=model_id, timesteps=horizon+2, load_checkpoint=False)
        model.load(chkpt)
    else:
        model = dmc.DynamicsModel(model_id=model_id, timesteps=horizon+2, load_checkpoint=True)
    # initialize the shared dktcache across MCTS trials
    if dktcache is None:
        dktcache = dict()
    
    acc = 0.0
    best_q = 0.0
    for i in xrange(n_trajectories):
        #print('traj i {}'.format(i))
        # create the model and simulators
        sim = s.copy()
        k, best_q_value = test_dkt_single(dgraph, sim, horizon, n_rollouts, model, r_type, dktcache, use_real)
        final_reward = np.sum(k)
        if r_type == SPARSE:
            final_reward = np.prod(k)
        acc += final_reward
        best_q += best_q_value
    return acc, best_q

def test_dkt(model_id, n_concepts, transition_after, horizon, n_rollouts, n_trajectories, r_type, use_real, chkpt=None):
    '''
    Test DKT+MCTS
    '''
    import concept_dependency_graph as cdg
    from simple_mdp import create_custom_dependency
    
    #learn_prob = 0.5
    n_jobs = 8
    traj_per_job =  n_trajectories // n_jobs

    #dgraph = create_custom_dependency()

    dgraph = cdg.ConceptDependencyGraph()
    dgraph.init_default_tree(n_concepts)

    #student = st.Student(n=n_concepts,p_trans_satisfied=learn_prob, p_trans_not_satisfied=0.0, p_get_ex_correct_if_concepts_learned=1.0)
    student2 = st.Student2(n_concepts, transition_after)
    test_student = student2
    
    test_student.reset()
    test_student.knowledge[0] = 1 # initialize the first concept to be known
    sim = st.StudentExactSim(test_student.copy(), dgraph)
    
    # create a shared dktcache across all processes
    dktcache_manager = mp.Manager()
    dktcache = dktcache_manager.dict()

    print('Testing model: {}'.format(model_id))
    print('horizon: {}'.format(horizon))
    print('rollouts: {}'.format(n_rollouts))

    accs = np.array(Parallel(n_jobs=n_jobs)(delayed(test_dkt_chunk)(traj_per_job, dgraph, sim, model_id, chkpt, horizon, n_rollouts, r_type, dktcache=dktcache, use_real=use_real) for _ in range(n_jobs)))
    results = np.sum(accs,axis=0) / (n_jobs * traj_per_job)
    avg_acc, avg_best_q = results[0], results[1]


    test_data = dg.generate_data(dgraph, student=test_student, n_students=1000, seqlen=horizon, policy='expert', filename=None, verbose=False)
    print('Average posttest true: {}'.format(expected_reward(test_data)))
    print('Average posttest mcts: {}'.format(avg_acc))
    print('Average best q: {}'.format(avg_best_q))
    return avg_acc, avg_best_q

def test_dkt_rme(model_id, n_rollouts, n_trajectories, r_type, dmcmodel, chkpt):
    '''
    Test DKT+MCTS where the real environment is a StudentDKTSim with a proxy DynamicsModel
    '''
    import concept_dependency_graph as cdg
    from simple_mdp import create_custom_dependency
    
    n_concepts = 4
    horizon = 6
    n_jobs = 8
    traj_per_job =  n_trajectories // n_jobs

    #dgraph = create_custom_dependency()
    dgraph = cdg.ConceptDependencyGraph()
    dgraph.init_default_tree(n_concepts)
    
    # create a shared dktcache across all processes
    dktcache_manager = mp.Manager()
    # for the MCTS model
    dktcache = dktcache_manager.dict()
    # for the real environment 
    dktsimcache = dktcache_manager.dict()
    
    # create the simulator
    dktsim = st.StudentDKTSim(dgraph, dmcmodel, dktsimcache)

    print('Testing proper RME model: {}'.format(model_id))
    print('horizon: {}'.format(horizon))
    print('rollouts: {}'.format(n_rollouts))

    accs = np.array(Parallel(n_jobs=n_jobs)(delayed(test_dkt_chunk)(traj_per_job, dgraph, dktsim, model_id, chkpt, horizon, n_rollouts, r_type, dktcache=dktcache, use_real=True) for _ in range(n_jobs)))
    results = np.sum(accs,axis=0) / (n_jobs * traj_per_job)
    avg_acc, avg_best_q = results[0], results[1]

    print('Average posttest mcts: {}'.format(avg_acc))
    print('Average best q: {}'.format(avg_best_q))
    return avg_acc, avg_best_q

def test_dkt_qval(model_id, n_concepts, transition_after, horizon, n_rollouts, r_type, chkpt=None):
    '''
    Test DKT+MCTS with loads of rollouts to estimate the initial qval
    '''
    import concept_dependency_graph as cdg
    from simple_mdp import create_custom_dependency
    
    #learn_prob = 0.5

    #dgraph = create_custom_dependency()

    dgraph = cdg.ConceptDependencyGraph()
    dgraph.init_default_tree(n_concepts)

    #student = st.Student(n=n_concepts,p_trans_satisfied=learn_prob, p_trans_not_satisfied=0.0, p_get_ex_correct_if_concepts_learned=1.0)
    student2 = st.Student2(n_concepts, transition_after)
    test_student = student2
    
    # load the model
    if chkpt is not None:
        model = dmc.DynamicsModel(model_id=model_id, timesteps=horizon, load_checkpoint=False)
        model.load(chkpt)
    else:
        model = dmc.DynamicsModel(model_id=model_id, timesteps=horizon, load_checkpoint=True)
    # initialize the dktcache to speed up DKT queries
    dktcache = dict()

    print('Testing model qval: {}'.format(model_id))
    print('horizon: {}'.format(horizon))
    print('rollouts: {}'.format(n_rollouts))
    
    # create the model and simulators
    stu = test_student.copy()
    stu.reset()
    stu.knowledge[0] = 1 # initialize the first concept to be known
    sim = st.StudentExactSim(stu, dgraph)

    # make the model
    dktmodel = dmc.RnnStudentSim(model)

    #rollout_policy = default_policies.immediate_reward
    rollout_policy = default_policies.RandomKStepRollOut(horizon+1)
    uct = MCTS(tree_policies.UCB1(1.41), rollout_policy,
               backups.monte_carlo) # 1.41 is sqrt (2), backups is from mcts.py

    root = StateNode(None, DKTState(dktmodel, sim, 1, horizon, r_type, dktcache, False))
    # run MCTS
    best_action = uct(root, n=n_rollouts)
    # get qvalue at the root
    qval = root.q
    
    six.print_('Initial qval: {}'.format(qval))

    return qval


def test_dkt_extract_policy(model_id, n_concepts, transition_after, horizon, n_rollouts, r_type, chkpt=None):
    '''
    Test DKT+MCTS to extract out the policy used in the real domain. Also return the qvals.
    '''
    import concept_dependency_graph as cdg
    from simple_mdp import create_custom_dependency
    
    #learn_prob = 0.5

    #dgraph = create_custom_dependency()

    dgraph = cdg.ConceptDependencyGraph()
    dgraph.init_default_tree(n_concepts)

    #student = st.Student(n=n_concepts,p_trans_satisfied=learn_prob, p_trans_not_satisfied=0.0, p_get_ex_correct_if_concepts_learned=1.0)
    student2 = st.Student2(n_concepts, transition_after)
    test_student = student2
    
    # load the model
    if chkpt is not None:
        model = dmc.DynamicsModel(model_id=model_id, timesteps=horizon, load_checkpoint=False)
        model.load(chkpt)
    else:
        model = dmc.DynamicsModel(model_id=model_id, timesteps=horizon, load_checkpoint=True)
    # initialize the dktcache to speed up DKT queries
    dktcache = dict()

    print('Extracting policy from model: {}'.format(model_id))
    print('horizon: {}'.format(horizon))
    print('rollouts: {}'.format(n_rollouts))
    
    # create the model and simulators
    stu = test_student.copy()
    stu.reset()
    stu.knowledge[0] = 1 # initialize the first concept to be known
    sim = st.StudentExactSim(stu, dgraph)

    # make the model
    dktmodel = dmc.RnnStudentSim(model)

    #rollout_policy = default_policies.immediate_reward
    rollout_policy = default_policies.RandomKStepRollOut(horizon+1)
    uct = MCTS(tree_policies.UCB1(1.41), rollout_policy,
               backups.monte_carlo) # 1.41 is sqrt (2), backups is from mcts.py

    root = StateNode(None, DKTState(dktmodel, sim, 1, horizon, r_type, dktcache, True))
    
    optpolicy = []
    qfunc = []
    
    for i in range(horizon):
        best_action = uct(root, n=n_rollouts)
        optpolicy.append(best_action.concept)
        qfunc.append([])
        for student_action in root.state.actions:
            qfunc[-1].append(root.children[student_action].q)
        # act in the real environment
        new_root = root.children[best_action].sample_state(real_world=True)
        new_root.parent = None # cutoff the rest of the tree
        root = new_root
    
    six.print_('Extracted policy: {}'.format(optpolicy))
    six.print_('Extracted q function: {}'.format(qfunc))

    return optpolicy, qfunc

class ExtractCallback(tflearn.callbacks.Callback):
    '''
    Used to get the training/validation losses after model.fit.
    '''
    def __init__(self):
        self.tstates = []
    def on_epoch_end(self, training_state):
        self.tstates.append(copy.copy(training_state))

def dkt_test_policy(model_id, horizon, n_trajectories, r_type, chkpt):
    '''
    Tests the uniformly random policy (behavior) for student2 n4 on the learned model.
    '''
    n_concepts = 4
    
    dgraph = cdg.ConceptDependencyGraph()
    dgraph.init_default_tree(n_concepts)
    
    student2 = st.Student2(n_concepts, transition_after)
    
    # load model from given file
    model = dmc.DynamicsModel(model_id=model_id, timesteps=horizon, load_checkpoint=False)
    model.load(chkpt)

    # create the model and simulators
    student = student2.copy()
    student.reset()
    student.knowledge[0] = 1 # initialize the first concept to be known
    sim = st.StudentExactSim(student, dgraph)
    
    # initialize the shared dktcache across the trials
    dktcache = dict()
    
    reward_acc = 0.0
    
    for t in six.moves.range(n_trajectories):
        # make the model
        rnnmodel = dmc.RnnStudentSim(model)

        curr_state = DKTState(rnnmodel, sim, 1, horizon, r_type, dktcache, False)
        all_actions = curr_state.actions
        for i in range(horizon):
            curr_state = curr_state.perform(random.choice(all_actions))
            reward_acc += curr_state.reward()
        #six.print_('Step: {}'.format(curr_state.step))
        #six.print_('Reward: {}'.format(curr_state.reward()))
        #six.print_('Reward Acc: {}'.format(reward_acc))
        #six.print_('Probs: {}'.format(curr_state.get_probs()))

    return reward_acc / n_trajectories

def dkt_test_policies_rme(model_id, n_trajectories, r_type, policies, chkpt):
    '''
    Tests a given open loop policy for student2 n4 on the learned model.
    '''
    
    horizon = 6
    n_concepts = 4
    
    dgraph = cdg.ConceptDependencyGraph()
    dgraph.init_default_tree(n_concepts)
    
    student2 = st.Student2(n_concepts, transition_after)
    
    # load model from given file
    model = dmc.DynamicsModel(model_id=model_id, timesteps=horizon, load_checkpoint=False)
    model.load(chkpt)

    # create the model and simulators
    student = student2.copy()
    student.reset()
    student.knowledge[0] = 1 # initialize the first concept to be known
    sim = st.StudentExactSim(student, dgraph)
    
    # initialize the shared dktcache across the trials
    dktcache = dict()
    
    num_policies = policies.shape[0]
    rewards = np.zeros((num_policies,))
    traj_per_policy = n_trajectories
    
    for pix in six.moves.range(num_policies):
        pol = policies[pix,:]
        reward_acc = 0.0
        for t in six.moves.range(traj_per_policy):
            # make the model
            rnnmodel = dmc.RnnStudentSim(model)

            curr_state = DKTState(rnnmodel, sim, 1, horizon, r_type, dktcache, False)
            all_actions = curr_state.actions
            for i in range(horizon):
                curr_state = curr_state.perform(all_actions[pol[i]])
            reward_acc += curr_state.reward()
        rewards[pix] = reward_acc / traj_per_policy
    
    return rewards

def dkt_test_mcts_proper_rme(model_id, n_rollouts, n_trajectories, r_type, envs, chkpt):
    '''
    Given a list of saved models as real environments, test the given saved model with MCTS.
    '''
    rewards = np.zeros((len(envs),))
    for i in six.moves.range(len(envs)):
        # create the proxy model and DKT sim
        sim_manager = dmc.DMCManager()
        sim_manager.start()
        dmcmodel = sim_manager.DynamicsModel(model_id=model_id,timesteps=6, load_checkpoint=False)
        #dmcmodel = dmc.DynamicsModel(model_id=model_id,timesteps=6, load_checkpoint=False)
        dmcmodel.load(envs[i])
        
        acc, qval = test_dkt_rme(model_id, n_rollouts, n_trajectories, r_type, dmcmodel, chkpt)
        rewards[i] = acc
    
    return rewards
    

def _dkt_train_models_chunk(params, runstartix, chunk_num_runs):
    '''
    Loads data and trains a batch of models.
    A batch is a continguous sequence of runs
    '''
    
    #six.print_('startix {} nruns {}'.format(runstartix,chunk_num_runs))
    
    train_losses = [[] for _ in six.moves.range(chunk_num_runs)]
    val_losses = [[] for _ in six.moves.range(chunk_num_runs)]
    
    #load data
    data = dataset_utils.load_data(filename='{}{}'.format(dg.SYN_DATA_DIR, params.datafile))
    input_data_, output_mask_, target_data_ = dataset_utils.preprocess_data_for_rnn(data)
    train_data = (input_data_[:,:,:], output_mask_[:,:,:], target_data_[:,:,:])
    
    for offset in six.moves.range(chunk_num_runs):
        r = runstartix + offset
        
        # new model instantiation
        dkt_model = dmc.DynamicsModel(model_id=params.model_id, timesteps=params.seqlen-1, dropout=params.dropout, load_checkpoint=False)
        
        epochs_trained = 0
        for ep in params.saved_epochs:
            print('=====================================')
            print('---------- Rep {:2d} Epoch {:2d} ----------'.format(r, ep))
            print('=====================================')
            
            # remember the epochs are given as zero-based
            epochs_to_train = ep+1 - epochs_trained
            assert epochs_to_train > 0
            
            # train
            ecall = ExtractCallback()
            dkt_model.train(train_data, n_epoch=epochs_to_train, callbacks=ecall, shuffle=params.shuffle, load_checkpoint=False)
            
            # save the checkpoint
            checkpoint_name = params.checkpoint_pat.format(params.run_name, r, ep)
            checkpoint_path = '{}/{}'.format(params.dir_name,checkpoint_name)
            dkt_model.save(checkpoint_path)
            
            # update stats
            train_losses[offset].extend([c.global_loss for c in ecall.tstates])
            val_losses[offset].extend([c.val_loss for c in ecall.tstates])
            
            # update epochs_trained
            epochs_trained = ep+1
    return (train_losses, val_losses)

def dkt_train_models(params):
    '''
    Trains a bunch of random restarts of models, checkpointed at various times
    '''
    
    # first try to create the checkpoint directory if it doesn't exist
    try:
        os.makedirs(params.dir_name)
    except:
        # do nothing if already exists
        pass
    
    train_losses = []
    val_losses = []
    
    n_jobs = min(5, params.num_runs)
    # need to be a multiple of number of jobs so I don't have to deal with uneven leftovers
    assert(params.num_runs % n_jobs == 0)
    runs_per_job = int(params.num_runs / n_jobs)
    
    losses = list(Parallel(n_jobs=n_jobs)(delayed(_dkt_train_models_chunk)(params,startix,runs_per_job) for startix in six.moves.range(0,params.num_runs,runs_per_job)))
    
    for tloss, vloss in losses:
        train_losses.extend(tloss)
        val_losses.extend(vloss)
    #six.print_((train_losses,val_losses))
    
    # save stats
    stats_path = '{}/{}'.format(params.dir_name,params.stat_name)
    np.savez(stats_path,tloss=train_losses, vloss=val_losses,eps=params.saved_epochs)


def dkt_test_models_mcts(trainparams,mctsparams):
    '''
    Given a set of runs, test the checkpointed models using MCTS
    '''
    
    scores = [[] for _ in six.moves.range(trainparams.num_runs)]
    qvals = [[] for _ in six.moves.range(trainparams.num_runs)]
    
    for r in six.moves.range(trainparams.num_runs):
        for ep in trainparams.saved_epochs:
            print('=====================================')
            print('---------- Rep {:2d} Epoch {:2d} ----------'.format(r, ep))
            print('=====================================')
            
            # load model from checkpoint
            checkpoint_name = trainparams.checkpoint_pat.format(trainparams.run_name, r, ep)
            checkpoint_path = '{}/{}'.format(trainparams.dir_name,checkpoint_name)
            
            # test dkt
            score, qval = test_dkt(
                trainparams.model_id, trainparams.n_concepts, trainparams.transition_after, 
                mctsparams.horizon, mctsparams.n_rollouts, mctsparams.n_trajectories,
                mctsparams.r_type, mctsparams.use_real, chkpt=checkpoint_path)
            
            # update stats
            scores[r].append(score)
            qvals[r].append(qval)
            
    
    # save stats
    mctsstat_name = mctsparams.stat_pat.format(trainparams.run_name)
    mctsstats_path = '{}/{}'.format(trainparams.dir_name,mctsstat_name)
    np.savez(mctsstats_path, scores=scores, qvals=qvals)

def _dkt_test_models_mcts_qval_single(trainparams,mctsparams,r,ep):
    print('=====================================')
    print('---------- Rep {:2d} Epoch {:2d} ----------'.format(r, ep))
    print('=====================================')

    # load model from checkpoint
    checkpoint_name = trainparams.checkpoint_pat.format(trainparams.run_name, r, ep)
    checkpoint_path = '{}/{}'.format(trainparams.dir_name,checkpoint_name)

    # test dkt
    qval = test_dkt_qval(
        trainparams.model_id, trainparams.n_concepts, trainparams.transition_after, mctsparams.horizon, mctsparams.initialq_n_rollouts, mctsparams.r_type, chkpt=checkpoint_path)
    return qval

def dkt_test_models_mcts_qval(trainparams,mctsparams):
    '''
    Given a set of runs, test the checkpointed models initial state's qval using loads of MCTS rollouts
    '''
    qvals = [[] for _ in six.moves.range(trainparams.num_runs)]
    
    flat_qvals = np.array(Parallel(n_jobs=8)(delayed(_dkt_test_models_mcts_qval_single)(trainparams,mctsparams,r,ep) for (r,ep) in itertools.product(six.moves.range(trainparams.num_runs), trainparams.saved_epochs)))
    
    flatix = 0
    for r in six.moves.range(trainparams.num_runs):
        for ep in trainparams.saved_epochs:
            qval = flat_qvals[flatix]
            # update stats
            qvals[r].append(qval)
            flatix += 1
    
    # save stats
    stat_name = mctsparams.initialq_pat.format(trainparams.run_name)
    mctsstats_path = '{}/{}'.format(trainparams.dir_name,stat_name)
    np.savez(mctsstats_path, qvals=qvals)

def dkt_test_models_extract_policy(trainparams,mctsparams):
    '''
    Given a set of runs, use MCTS to extrac their policy
    '''
    optpolicies = [[] for _ in six.moves.range(trainparams.num_runs)]
    qfuncs = [[] for _ in six.moves.range(trainparams.num_runs)]
    
    for r in six.moves.range(trainparams.num_runs):
        for ep in trainparams.saved_epochs:
            print('=====================================')
            print('---------- Rep {:2d} Epoch {:2d} ----------'.format(r, ep))
            print('=====================================')
            
            # load model from checkpoint
            checkpoint_name = trainparams.checkpoint_pat.format(trainparams.run_name, r, ep)
            checkpoint_path = '{}/{}'.format(trainparams.dir_name,checkpoint_name)
            
            # test dkt
            optpolicy, qfunc = test_dkt_extract_policy(
                trainparams.model_id, trainparams.n_concepts, trainparams.transition_after, mctsparams.horizon, mctsparams.policy_n_rollouts, mctsparams.r_type, chkpt=checkpoint_path)
            
            # update stats
            optpolicies[r].append(optpolicy)
            qfuncs[r].append(qfunc)
            
    
    # save stats
    stat_name = mctsparams.optpolicy_pat.format(trainparams.run_name)
    mctsstats_path = '{}/{}'.format(trainparams.dir_name,stat_name)
    np.savez(mctsstats_path, opts=optpolicies, qs=qfuncs)

def dkt_test_models_policy(trainparams,mctsparams):
    '''
    Given a set of runs, test the uniform random policy (behavior policy) using the checkpointed models 
    '''
    
    rewards = [[] for _ in six.moves.range(trainparams.num_runs)]
    
    for r in six.moves.range(trainparams.num_runs):
        for ep in trainparams.saved_epochs:
            print('=====================================')
            print('---------- Rep {:2d} Epoch {:2d} ----------'.format(r, ep))
            print('=====================================')
            
            # load model from checkpoint
            checkpoint_name = trainparams.checkpoint_pat.format(trainparams.run_name, r, ep)
            checkpoint_path = '{}/{}'.format(trainparams.dir_name,checkpoint_name)
            
            # test dkt
            reward_avg = dkt_test_policy(trainparams.model_id, trainparams.seqlen, mctsparams.n_trajectories, mctsparams.r_type, checkpoint_path)
            #six.print_('Reward avg: {}'.format(reward_avg))
            
            # update stats
            rewards[r].append(reward_avg)
            
    
    # save stats
    policystat_name = mctsparams.policy_pat.format(trainparams.run_name)
    policystats_path = '{}/{}'.format(trainparams.dir_name,policystat_name)
    np.savez(policystats_path, rewards=rewards)

def dkt_test_models_rme(trainparams,mctsparams,policies):
    '''
    Given a set of runs, test a given set of policies corresponding to a robust matrix evaluation.
    '''
    
    rewards = np.zeros((trainparams.num_runs, len(policies)))
    
    for r in six.moves.range(trainparams.num_runs):
        for ep in trainparams.saved_epochs:
            print('=====================================')
            print('---------- Rep {:2d} Epoch {:2d} ----------'.format(r, ep))
            print('=====================================')
            
            # load model from checkpoint
            checkpoint_name = trainparams.checkpoint_pat.format(trainparams.run_name, r, ep)
            checkpoint_path = '{}/{}'.format(trainparams.dir_name,checkpoint_name)
            
            # test dkt
            curr_rewards = dkt_test_policies_rme(trainparams.model_id, mctsparams.rme_n_trajectories, mctsparams.r_type, policies, checkpoint_path)
            
            # update stats
            rewards[r,:] = curr_rewards
    
    # save stats
    rmestat_name = mctsparams.rme_pat.format(trainparams.run_name)
    rmestats_path = '{}/{}'.format(trainparams.dir_name,rmestat_name)
    np.savez(rmestats_path, evals=rewards)

def dkt_test_models_proper_rme(trainparams,mctsparams,envs):
    '''
    Given a set of runs, test a given set of models as the real environment corresponding to a robust matrix evaluation.
    '''
    
    rewards = np.zeros((trainparams.num_runs, len(envs)))
    
    for r in six.moves.range(trainparams.num_runs):
        for ep in trainparams.saved_epochs:
            print('=====================================')
            print('---------- Rep {:2d} Epoch {:2d} ----------'.format(r, ep))
            print('=====================================')
            
            # load model from checkpoint
            checkpoint_name = trainparams.checkpoint_pat.format(trainparams.run_name, r, ep)
            checkpoint_path = '{}/{}'.format(trainparams.dir_name,checkpoint_name)
            
            curr_rewards = dkt_test_mcts_proper_rme(
                trainparams.model_id, mctsparams.rme_n_rollouts, mctsparams.rme_n_trajectories, mctsparams.r_type, envs, checkpoint_path)
            
            # update stats
            rewards[r,:] = curr_rewards
    
    # save stats
    rmestat_name = mctsparams.rmeproper_pat.format(trainparams.run_name)
    rmestats_path = '{}/{}'.format(trainparams.dir_name,rmestat_name)
    np.savez(rmestats_path, evals=rewards)

if __name__ == '__main__':
    starttime = time.time()

    np.random.seed()
    random.seed()
    
    ############################################################################
    # testing MCTS and making sure it works when given the true model
    #test_student_exact()
    ############################################################################
    
    
    ############################################################################
    # General tests where I train a bunch of models and save them
    # then I run analysis on the saved models
    ############################################################################
    
    class TrainParams(object):
        '''
        Parameters for training models. These are the ones corresponding to student2 with 4 skills where the optimal policy takes 6 steps.
        '''
        def __init__(self, rname, nruns, model_id, seqlen, saved_epochs, dropout=1.0):
            self.model_id = model_id
            self.n_concepts = 5
            self.transition_after = True
            self.dropout = dropout
            self.shuffle = True
            self.seqlen = seqlen
            self.datafile = 'test2a-w{}-n100000-l{}-random.pickle'.format(self.n_concepts, self.seqlen)
            # which epochs (zero-based) to save, the last saved epoch is the total epoch
            self.saved_epochs = saved_epochs
            # name of these runs, which should be unique to one call to train models (unless you want to overwrite)
            self.run_name = rname
            # how many runs
            self.num_runs = nruns

            # these names are derived from above and should not be touched generally
            # folder to put the checkpoints into
            self.dir_name = 'experiments/{}-dropout{}-shuffle{}-data-{}'.format(
                self.model_id,int(self.dropout*10),int(self.shuffle),self.datafile)
            # pattern for the checkpoints
            self.checkpoint_pat = 'checkpoint-{}{}-epoch{}'
            # stat file
            self.stat_name = 'stats-{}'.format(self.run_name)
    
    class TrainParams2(object):
        '''
        Parameters for training models. These correspond to student2 with 2 skills, and the optimal policy is 2 steps.
        '''
        def __init__(self, rname, nruns, model_id, saved_epochs):
            #self.model_id = 'test2_model2simple_tiny'
            #self.model_id = 'test2_model2_tiny'
            #self.model_id = 'test2_model2gru_tiny'
            self.model_id = model_id
            self.n_concepts = 2
            self.dropout = 1.0
            self.shuffle = False
            self.seqlen = 3 # have tried length 2 and length 3
            self.datafile = 'test2-n10000-l{}-random.pickle'.format(self.seqlen)
            # which epochs (zero-based) to save, the last saved epoch is the total epoch
            # for length 2
            # 54, 46 simple, 43 gru, 20 for earlier for simple and gru, 30 for earlier for lstm
            # with binary crossentropy, 40, 40 simple, 40 gru (maybe 30 if you feel like it)
            # for length 3
            self.saved_epochs = saved_epochs
            # name of these runs, which should be unique to one call to train models (unless you want to overwrite)
            self.run_name = rname
            # how many runs
            self.num_runs = nruns

            # these names are derived from above and should not be touched generally
            # folder to put the checkpoints into
            self.dir_name = 'experiments/{}-dropout{}-shuffle{}-data-{}'.format(
                self.model_id,int(self.dropout*10),int(self.shuffle),self.datafile)
            # pattern for the checkpoints
            self.checkpoint_pat = 'checkpoint-{}{}-epoch{}'
            # stat file
            self.stat_name = 'stats-{}'.format(self.run_name)
        
    #----------------------------------------------------------------------
    # train and checkpoint the models
    
    # student2 4 skills dropout 8 data - don't forget to change the dropout and saved_epochs
    #cur_train = [TrainParams('runA', 20), TrainParams('runC', 30), TrainParams('runD', 50)]
    #cur_train = [TrainParams('runA', 20)]
    #cur_train = [TrainParams('runB', 30)]
    
    # student2 4 skills dropout 10 data - don't forget to change the dropout and saved_epochs
    #cur_train = [TrainParams('runA', 10)]
    #cur_train = [TrainParams('runA', 10), TrainParams('runB', 90)]
    
    # student2 with 2 skills training
    
    # train binary crossentropy OLD
    #cur_train = [TrainParams2('runBinCE-A',50)]
    
    # trying to determine when to stop training
    #cur_train = [TrainParams2('runA',20,'test2_model2_tiny'), TrainParams2('runA',20,'test2_model2simple_tiny'), TrainParams2('runA',20,'test2_model2gru_tiny')]
    
    # train 50 of each architecture
    #cur_train = [TrainParams2('runB',50,'test2_model2_tiny', [40]), TrainParams2('runB',50,'test2_model2simple_tiny',[25]), TrainParams2('runB',50,'test2_model2gru_tiny',[25])]
    
    # student2 4 skills with training trajectory length 7
    # first find try to find when to stop
    #cur_train = [TrainParams('runA',10,'test2_model_small', [20]), TrainParams('runA',10,'test2_modelsimple_small',[20]), TrainParams('runA',10,'test2_modelgru_small',[20])]
    
    # found epoch 12 is good to stop, so now learn 50 models each
    #cur_train = [TrainParams('runB',50,'test2_model_small', [12]), TrainParams('runB',50,'test2_modelsimple_small',[12]),TrainParams('runB',50,'test2_modelgru_small',[12])]
    
    # student2 4 skills with training trajectory length egreedy 0.30
    # first try to find when to stop
    #cur_train = [TrainParams('runA',10,'test2_model_small', [20]), TrainParams('runA',10,'test2_modelsimple_small',[20]), TrainParams('runA',10,'test2_modelgru_small',[20])]
    
    # found stopping epochs, so now learn 50 models each
    #cur_train = [TrainParams('runB',50,'test2_model_small', [9]), TrainParams('runB',50,'test2_modelsimple_small',[8]),TrainParams('runB',50,'test2_modelgru_small',[6])]
    
    # student2 4 skills with training trajectory length 7, random behavior policy, mid-size model
    # trying to determine when to stop
    #cur_train = [TrainParams('runA',10,'test2_model_mid', [20]), TrainParams('runA',10,'test2_modelsimple_mid',[20]), TrainParams('runA',10,'test2_modelgru_mid',[20])]
    #cur_train = [TrainParams('runB',50,'test2_model_mid', [6]), TrainParams('runB',50,'test2_modelsimple_mid',[5]),TrainParams('runB',50,'test2_modelgru_mid',[4])]
    
    # student2 4 skills with training trajectory length 7, random behavior policy, model student2a domain
    # mid-size models
    # trying to determine when to stop
    #cur_train = [TrainParams('runA',10,'test2_model_mid', [20]), TrainParams('runA',10,'test2_modelsimple_mid',[20]), TrainParams('runA',10,'test2_modelgrusimple_mid',[20])]
    
    #cur_train = [TrainParams('runA',50,'test2_model_mid', [11]), TrainParams('runA',50,'test2_modelsimple_mid',[11]), TrainParams('runA',50,'test2_modelgrusimple_mid',[7])]
    
    #small-size models
    # trying to determine when to stop
    #cur_train = [TrainParams('runA',10,'test2_modelsimple_small',[20]), TrainParams('runA',10,'test2_modelgrusimple_small',[20])]
    
    # testing first with 20 points
    #cur_train = [TrainParams('runB',20,'test2_modelsimple_small',[20]), TrainParams('runB',20,'test2_modelgrusimple_small',[14])]
    
    # student2 4 skills with training trajectory length 5, random behavior policy, model student2a domain
    # small-size models
    # try only the two simple architectures
    # first determine when to stop
    #cur_train = [TrainParams('runA',10,'test2_modelsimple_small',[20]), TrainParams('runA',10,'test2_modelgrusimple_small',[20])]
    
    # testing with 50 points
    #cur_train = [TrainParams('runB',50,'test2_modelsimple_small',[14]), TrainParams('runB',50,'test2_modelgrusimple_small',[18])]
    
    # mid-size models now, hopefully they do better
    # first determine when to stop
    #cur_train = [TrainParams('runA',10,'test2_modelsimple_mid',[20]), TrainParams('runA',10,'test2_modelgrusimple_mid',[20])]
    
    # testing first with 20 points
    #cur_train = [TrainParams('runB',20,'test2_modelsimple_mid',[12]), TrainParams('runB',20,'test2_modelgrusimple_mid',[8])]
    
    # trying out 2 more architectures of double lstm which seem like it might be good at generalization
    # first determine when to stop
    #cur_train = [TrainParams('runA',10,'test2_model_small',[20]), TrainParams('runA',10,'test2_model_mid',[20])]
    
    # first test with 20 points
    #cur_train = [TrainParams('runB',20,'test2_model_small',[18]), TrainParams('runB',20,'test2_model_mid',[11])]
    
    # pick the single LSTM model small and try out further experiments
    #cur_train = [TrainParams('runB',50,'test2_modelsimple_small',[14])]
    
    # pick the single LSTM mid size model and try further experiments
    # use dropout, and first try to find stopping point, and use shuffle to smooth training
    #cur_train = [TrainParams('runA',10,'test2_modelsimple_mid',[50], dropout=0.8)]
    
    # first try testing 20 models
    #cur_train = [TrainParams('runB',20,'test2_modelsimple_mid',[30], dropout=0.8)]
    
    # student2 4 skills with training trajectory length 6, random, student2a domain
    # midsize model
    # first try to find when to stop
    #cur_train = [TrainParams('runA',10,'test2_modelsimple_mid',[20])]
    #cur_train = [TrainParams('runA',10,'test2_modelgrusimple_mid',[20])]
    
    # first try testing 20 models
    #cur_train = [TrainParams('runB',20,'test2_modelsimple_mid',[12])]
    
    # now continue with 30 models
    #cur_train = [TrainParams('runC',30,'test2_modelsimple_mid',[12])]
    
    # try 50 with simple gru
    #cur_train = [TrainParams('runB',50,'test2_modelgrusimple_mid',[9])]
    
    # student2a 5 skills with training trajectory, random
    # go back to mean squared loss and tuned learning rate of 0.01
    # everything needs like 40 epochs to train, so directly go training
    # use gru simple and do dropout and without
    # dropout seems to suck though
    #cur_train = [TrainParams('runlr01A',20,'test2w5_modelgrusimple_mid',7,[40],dropout=0.8)]
    # length 7
    #cur_train = [TrainParams('runlr01A',20,'test2w5_modelgrusimple_mid',7,[40]), TrainParams('runlr01B',30,'test2w5_modelgrusimple_mid',7,[40])]
    # length 6
    cur_train = [TrainParams('runlr01A',30,'test2w5_modelgrusimple_mid',6,[50]),TrainParams('runlr01B',20,'test2w5_modelgrusimple_mid',6,[50])]
    
    for ct in cur_train:
        pass
        #dkt_train_models(ct)
    #----------------------------------------------------------------------
    
    class TestParams:
        '''
        Parameters for testing models with MCTS/policies. For testing student2 with 4 skills.
        '''
        def __init__(self, use_real=True):
            self.r_type = SPARSE
            self.n_rollouts = 20000
            self.n_trajectories = 8
            self.use_real = use_real
            self.horizon = 8
            
            # for testing initialq values
            self.initialq_n_rollouts = 200000
            
            # for extracting a policy
            self.policy_n_rollouts = 20000
            
            # for rme
            self.rme_n_rollouts = 1000
            self.rme_n_trajectories = 100
            
            # below are generated values from above
            # stat filename pattern
            self.stat_pat = 'mcts-rtype{}-rollouts{}-trajectories{}-real{}-{{}}'.format(
                self.r_type, self.n_rollouts, self.n_trajectories, int(self.use_real))
            # stat filename for policy testing
            self.policy_pat = 'policies-rtype{}-trajectories{}-{{}}'.format(
                self.r_type, self.n_trajectories)
            # state filename for initial qval teseting
            self.initialq_pat = 'initialq-rtype{}-rollouts{}-{{}}'.format(
                self.r_type, self.initialq_n_rollouts)
            
            # stat for extracting a policy
            self.optpolicy_pat = 'optpolicy-rtype{}-rollouts{}-{{}}'.format(
                self.r_type, self.policy_n_rollouts)
            
            # stat for robust matrix evaluation
            self.rme_pat = 'rme-rtype{}-trajectories{}-{{}}'.format(
                self.r_type, self.rme_n_trajectories)
            
            # stat for robust matrix evaluation
            self.rmeproper_pat = 'rmeproper-rtype{}-rollouts{}-trajectories{}-{{}}'.format(
                self.r_type, self.rme_n_rollouts, self.rme_n_trajectories)
    
    class TestParams2:
        '''
        Parameters for testing models with MCTS/policies. For testing student2 with 2 skills.
        '''
        def __init__(self, use_real=True):
            self.r_type = SPARSE
            self.n_rollouts = 1000
            self.n_trajectories = 100
            self.use_real = use_real
            self.horizon = 2
            
            # for testing initialq values
            self.initialq_n_rollouts = 100000
            
            # for extracting a policy
            self.policy_n_rollouts = 20000
            
            # for rme
            self.rme_n_rollouts = 1000
            self.rme_n_trajectories = 100
            
            # below are generated values from above
            # stat filename pattern
            self.stat_pat = 'mcts-rtype{}-rollouts{}-trajectories{}-real{}-{{}}'.format(
                self.r_type, self.n_rollouts, self.n_trajectories, int(self.use_real))
            # stat filename for policy testing
            self.policy_pat = 'policies-rtype{}-trajectories{}-{{}}'.format(
                self.r_type, self.n_trajectories)
            # state filename for initial qval teseting
            self.initialq_pat = 'initialq-rtype{}-rollouts{}-{{}}'.format(
                self.r_type, self.initialq_n_rollouts)
            
            # stat for extracting a policy
            self.optpolicy_pat = 'optpolicy-rtype{}-rollouts{}-{{}}'.format(
                self.r_type, self.policy_n_rollouts)
            
            # stat for robust matrix evaluation
            self.rme_pat = 'rme-rtype{}-trajectories{}-{{}}'.format(
                self.r_type, self.rme_n_trajectories)
            
            # stat for robust matrix evaluation
            self.rmeproper_pat = 'rmeproper-rtype{}-rollouts{}-trajectories{}-{{}}'.format(
                self.r_type, self.rme_n_rollouts, self.rme_n_trajectories)
    
    #----------------------------------------------------------------------
    # test the saved models
    # don't train and test at the same time, alternate between them
    
    # read the optpolicies from data
    if False:
        data1 = np.load('experiments/test2_model_small-dropout10-shuffle0-data-test2-n100000-l5-random.pickle/optpolicy-rtype1-rollouts10000-runA.npz')
        data2 = np.load('experiments/test2_model_small-dropout10-shuffle0-data-test2-n100000-l5-random.pickle/optpolicy-rtype1-rollouts10000-runB.npz')
        opts = np.vstack([data1['opts'],data2['opts']])[:,0,:]
        
        data61 = np.load('experiments/test2_model_small-dropout8-shuffle0-data-test2-n100000-l5-random.pickle/optpolicy-rtype1-rollouts10000-runA.npz')
        data62 = np.load('experiments/test2_model_small-dropout8-shuffle0-data-test2-n100000-l5-random.pickle/optpolicy-rtype1-rollouts10000-runC.npz')
        data63 = np.load('experiments/test2_model_small-dropout8-shuffle0-data-test2-n100000-l5-random.pickle/optpolicy-rtype1-rollouts10000-runD.npz')
        opts2 = np.vstack([data61['opts'],data62['opts'],data63['opts']])[:,0,:]
    # the models to use as real environments for proper rme
    if False:
        envs = []
        for ct in cur_train:
            for r in six.moves.range(ct.num_runs):
                for ep in ct.saved_epochs:
                    # load model from checkpoint
                    checkpoint_name = ct.checkpoint_pat.format(ct.run_name, r, ep)
                    checkpoint_path = '{}/{}'.format(ct.dir_name,checkpoint_name)
                    envs.append(checkpoint_path)
        six.print_('\n'.join(envs))

    
    tp = TestParams(use_real=False)
    for ct in cur_train:
        pass
        dkt_test_models_mcts(ct,tp)
        #dkt_test_models_mcts_qval(ct,tp)
        #dkt_test_models_extract_policy(ct,tp)
        #dkt_test_models_proper_rme(ct,tp,envs)
        #dkt_test_models_policy(ct,tp)
    #----------------------------------------------------------------------
    
    ############################################################################
    ############################################################################

    endtime = time.time()
    print('Time elapsed {}s'.format(endtime-starttime))

