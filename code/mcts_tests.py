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

def test_student_exact_single_greedy(dgraph, stud, horizon, n_rollouts):
    '''
    Performs a single trajectory with the greedy 1-step policy and returns the final true student knowlegde.
    '''
    n_concepts = dgraph.n

    # create the model and simulators
    s = stud.copy()
    s.reset()
    s.knowledge[0] = 1 # initialize the first concept to be known
    sim = st.StudentExactSim(s, dgraph)
    model = sim.copy()

    greedy = ExactGreedyPolicy(model, sim)

    for i in range(horizon):
        best_action = greedy.best_greedy_action(n_rollouts)

        # debug check for whether action is optimal
        if True:
            opt_acts = compute_optimal_actions(sim.dgraph, sim.student.knowledge)
            is_opt = best_action in opt_acts
            if not is_opt:
                print('ERROR {} executed non-optimal action {}'.format(sim.student.knowledge, best_action))
                # now let's print out even more debugging information
                #breadth_first_search(root, fnc=debug_visiter)
                #return None

        # act in the real environment
        greedy.advance(best_action)
    return sim.student.knowledge


def test_student_exact_chunk(n_trajectories, dgraph, student, horizon, n_rollouts, use_greedy, r_type):
    '''
    Runs a bunch of trajectories and returns the avg posttest score.
    For parallelization to run in a separate thread/process.
    '''
    acc = 0.0
    for i in xrange(n_trajectories):
        print('traj i {}'.format(i))
        if use_greedy:
            k = test_student_exact_single_greedy(dgraph, student, horizon, n_rollouts)
        else:
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
    use_greedy = False
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
    student2 = st.Student2(n_concepts)
    test_student = student2

    accs = Parallel(n_jobs=n_jobs)(delayed(test_student_exact_chunk)(traj_per_job, dgraph, test_student, horizon, n_rollouts, use_greedy, sparse_r) for _ in range(n_jobs))
    avg = sum(accs) / (n_jobs * traj_per_job)

    test_data = dg.generate_data(dgraph, student=test_student, n_students=1000, seqlen=horizon, policy='expert', filename=None, verbose=False)
    print('Number of jobs {}'.format(n_jobs))
    print('Trajectory per job {}'.format(traj_per_job))
    print('Use greedy? {}'.format(use_greedy))
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

def test_dkt_chunk(n_trajectories, dgraph, s, model_id, chkpt, horizon, n_rollouts, use_greedy, r_type, dktcache=None, use_real=True):
    '''
    Runs a bunch of trajectories and returns the avg posttest score.
    For parallelization to run in a separate thread/process.
    '''
    # load the model
    if chkpt is not None:
        model = dmc.DynamicsModel(model_id=model_id, timesteps=horizon, load_checkpoint=False)
        model.load(chkpt)
    else:
        model = dmc.DynamicsModel(model_id=model_id, timesteps=horizon, load_checkpoint=True)
    # initialize the shared dktcache across MCTS trials
    if dktcache is None:
        dktcache = dict()
    
    acc = 0.0
    best_q = 0.0
    for i in xrange(n_trajectories):
        #print('traj i {}'.format(i))
        # create the model and simulators
        sim = s.copy()
        if not use_greedy:
            k, best_q_value = test_dkt_single(dgraph, sim, horizon, n_rollouts, model, r_type, dktcache, use_real)
        else:
            # This branch is DEPRECATED
            assert(False)
        acc += np.mean(k)
        best_q += best_q_value
    return acc, best_q

def test_dkt(model_id, n_rollouts, n_trajectories, r_type, use_real, chkpt=None):
    '''
    Test DKT+MCTS
    '''
    import concept_dependency_graph as cdg
    from simple_mdp import create_custom_dependency
    
    n_concepts = 4
    use_greedy = False
    #learn_prob = 0.5
    horizon = 6
    n_jobs = 8
    traj_per_job =  n_trajectories // n_jobs

    #dgraph = create_custom_dependency()

    dgraph = cdg.ConceptDependencyGraph()
    dgraph.init_default_tree(n_concepts)

    #student = st.Student(n=n_concepts,p_trans_satisfied=learn_prob, p_trans_not_satisfied=0.0, p_get_ex_correct_if_concepts_learned=1.0)
    student2 = st.Student2(n_concepts)
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

    accs = np.array(Parallel(n_jobs=n_jobs)(delayed(test_dkt_chunk)(traj_per_job, dgraph, sim, model_id, chkpt, horizon, n_rollouts, use_greedy, r_type, dktcache=dktcache, use_real=use_real) for _ in range(n_jobs)))
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

    accs = np.array(Parallel(n_jobs=n_jobs)(delayed(test_dkt_chunk)(traj_per_job, dgraph, dktsim, model_id, chkpt, horizon, n_rollouts, False, r_type, dktcache=dktcache, use_real=True) for _ in range(n_jobs)))
    results = np.sum(accs,axis=0) / (n_jobs * traj_per_job)
    avg_acc, avg_best_q = results[0], results[1]

    print('Average posttest mcts: {}'.format(avg_acc))
    print('Average best q: {}'.format(avg_best_q))
    return avg_acc, avg_best_q

def test_dkt_qval(model_id, n_rollouts, r_type, chkpt=None):
    '''
    Test DKT+MCTS with loads of rollouts to estimate the initial qval
    '''
    import concept_dependency_graph as cdg
    from simple_mdp import create_custom_dependency
    
    n_concepts = 4
    use_greedy = False
    #learn_prob = 0.5
    horizon = 6

    #dgraph = create_custom_dependency()

    dgraph = cdg.ConceptDependencyGraph()
    dgraph.init_default_tree(n_concepts)

    #student = st.Student(n=n_concepts,p_trans_satisfied=learn_prob, p_trans_not_satisfied=0.0, p_get_ex_correct_if_concepts_learned=1.0)
    student2 = st.Student2(n_concepts)
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


def test_dkt_extract_policy(model_id, n_rollouts, r_type, chkpt=None):
    '''
    Test DKT+MCTS to extract out the policy used in the real domain. Also return the qvals.
    '''
    import concept_dependency_graph as cdg
    from simple_mdp import create_custom_dependency
    
    n_concepts = 4
    use_greedy = False
    #learn_prob = 0.5
    horizon = 6

    #dgraph = create_custom_dependency()

    dgraph = cdg.ConceptDependencyGraph()
    dgraph.init_default_tree(n_concepts)

    #student = st.Student(n=n_concepts,p_trans_satisfied=learn_prob, p_trans_not_satisfied=0.0, p_get_ex_correct_if_concepts_learned=1.0)
    student2 = st.Student2(n_concepts)
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

def staggered_dkt_training(filename, model_id, seqlen, dropout, shuffle, reps, total_epochs, epochs_per_iter, sig_start, return_channel):
    '''
    DEPRECATED
    This is supposed to be used in a separate process that trains a DKT epoch by epoch.
    It waits for the parent process to send an Event signal before each training epoch.
    It returns an intermediate value when done.
    It also returns the training/validation losses.
    '''
    data = dataset_utils.load_data(filename='{}{}'.format(dg.SYN_DATA_DIR, filename))
    input_data_, output_mask_, target_data_ = dataset_utils.preprocess_data_for_rnn(data)
    train_data = (input_data_[:,:,:], output_mask_[:,:,:], target_data_[:,:,:])
    
    for r in xrange(reps):
        dmodel = dmc.DynamicsModel(model_id=model_id, timesteps=seqlen, dropout=dropout, load_checkpoint=False)
        for ep in xrange(0,total_epochs,epochs_per_iter):
            sig_start.wait()
            sig_start.clear()
            #print('Training rep {} epoch {}'.format(r, ep))
            ecall = ExtractCallback()
            dmodel.train(train_data, n_epoch=epochs_per_iter, callbacks=ecall, shuffle=shuffle, load_checkpoint=False)
            return_channel.send(([c.global_loss for c in ecall.tstates],[c.val_loss for c in ecall.tstates]))

def test_dkt_early_stopping():
    '''
    DEPRECATED
    Performs an experiment where every few epochs of training we test with MCTS, and this is repeated.
    '''
    model_id = 'test2_model_small'
    r_type = SEMISPARSE
    dropout = 1.0
    shuffle = False
    n_rollouts = 1000
    n_trajectories = 100
    seqlen = 5
    filename = 'test2-n100000-l{}-random.pickle'.format(seqlen) # < 6 is already no full mastery

    total_epochs = 14
    epochs_per_iter = 14
    reps = 40
    
    sig_start = mp.Event()
    (p_ch, c_ch) = mp.Pipe()
    training_process = mp.Process(target=staggered_dkt_training,
                                  args=(filename, model_id, seqlen, dropout, shuffle, reps, total_epochs, epochs_per_iter, sig_start, c_ch))
    
    losses = []
    val_losses = []
    score_eps = []
    scores = []
    best_qs = []
    
    training_process.start()
    for r in xrange(reps):
        losses.append(list())
        val_losses.append(list())
        score_eps.append(list())
        scores.append(list())
        best_qs.append(list())
        for ep in xrange(0,total_epochs,epochs_per_iter):
            print('=====================================')
            print('---------- Rep {:2d} Epoch {:2d} ----------'.format(r+1, ep+epochs_per_iter))
            print('=====================================')
            sig_start.set() # signal the training to perform one epoch
            ret_val = p_ch.recv() # wait until its done
            losses[r].extend(ret_val[0])
            val_losses[r].extend(ret_val[1])
            
            # now compute the policy estimate
            score_eps[r].append(ep+epochs_per_iter)
            score, _ = test_dkt(model_id, n_rollouts, n_trajectories, r_type, True)
            _, best_q = test_dkt(model_id, n_rollouts, n_trajectories, r_type, False)
            scores[r].append(score)
            best_qs[r].append(best_q)
    training_process.join() # finish up
    
    np.savez("earlystopping",losses=losses, vals=val_losses, eps=score_eps, scores=scores, qs=best_qs)

def dkt_test_policy(model_id, horizon, n_trajectories, r_type, chkpt):
    '''
    Tests the uniformly random policy (behavior) for student2 n4 on the learned model.
    '''
    n_concepts = 4
    
    dgraph = cdg.ConceptDependencyGraph()
    dgraph.init_default_tree(n_concepts)
    
    student2 = st.Student2(n_concepts)
    
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
    
    student2 = st.Student2(n_concepts)
    
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
    
    # total number of epochs to train for, should be passed in as the last epoch to be saved
    total_epochs = params.saved_epochs[-1]
    
    train_losses = [[] for _ in six.moves.range(params.num_runs)]
    val_losses = [[] for _ in six.moves.range(params.num_runs)]
    
    #load data
    data = dataset_utils.load_data(filename='{}{}'.format(dg.SYN_DATA_DIR, params.datafile))
    input_data_, output_mask_, target_data_ = dataset_utils.preprocess_data_for_rnn(data)
    train_data = (input_data_[:,:,:], output_mask_[:,:,:], target_data_[:,:,:])
    
    for r in six.moves.range(params.num_runs):
        # new model instantiation
        dkt_model = dmc.DynamicsModel(model_id=params.model_id, timesteps=params.seqlen, dropout=params.dropout, load_checkpoint=False)
        
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
            train_losses[r].extend([c.global_loss for c in ecall.tstates])
            val_losses[r].extend([c.val_loss for c in ecall.tstates])
            
            # update epochs_trained
            epochs_trained = ep+1
    
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
                trainparams.model_id, mctsparams.n_rollouts, mctsparams.n_trajectories,
                mctsparams.r_type, mctsparams.use_real, chkpt=checkpoint_path)
            
            # update stats
            scores[r].append(score)
            qvals[r].append(qval)
            
    
    # save stats
    mctsstat_name = mctsparams.stat_pat.format(trainparams.run_name)
    mctsstats_path = '{}/{}'.format(trainparams.dir_name,mctsstat_name)
    np.savez(mctsstats_path, scores=scores, qvals=qvals)

def dkt_test_models_mcts_qval(trainparams,mctsparams):
    '''
    Given a set of runs, test the checkpointed models initial state's qval using loads of MCTS rollouts
    '''
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
            qval = test_dkt_qval(
                trainparams.model_id, mctsparams.initialq_n_rollouts, mctsparams.r_type, chkpt=checkpoint_path)
            
            # update stats
            qvals[r].append(qval)
            
    
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
                trainparams.model_id, mctsparams.policy_n_rollouts, mctsparams.r_type, chkpt=checkpoint_path)
            
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
        Parameters for training models
        '''
        def __init__(self, rname, nruns):
            self.model_id = 'test2_model_small'
            self.dropout = 0.8
            self.shuffle = False
            self.seqlen = 5
            self.datafile = 'test2-n100000-l{}-random.pickle'.format(self.seqlen) # < 6 is already no full mastery
            # which epochs (zero-based) to save, the last saved epoch is the total epoch
            self.saved_epochs = [23]
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
    
    # dropout 8 data - don't forget to change the dropout and saved_epochs
    cur_train = [TrainParams('runA', 20), TrainParams('runC', 30), TrainParams('runD', 50)]
    #cur_train = [TrainParams('runA', 20)]
    #cur_train = [TrainParams('runB', 30)]
    
    # dropout 10 data - don't forget to change the dropout and saved_epochs
    #cur_train = [TrainParams('runA', 10)]
    #cur_train = [TrainParams('runA', 10), TrainParams('runB', 90)]
    
    #dkt_train_models(TrainParams())
    #----------------------------------------------------------------------
    
    class TestParams:
        '''
        Parameters for testing models with MCTS/policies
        '''
        def __init__(self, use_real=True):
            self.r_type = SPARSE
            self.n_rollouts = 3000
            self.n_trajectories = 400
            self.use_real = use_real
            
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

    
    tp = TestParams()
    #dkt_test_models_mcts(TrainParams(),TestParams(use_real=True))
    #dkt_test_models_policy(TrainParams(),TestParams())
    #dkt_test_models_mcts_qval(TrainParams(),TestParams())
    for ct in cur_train:
        pass
        #dkt_test_models_rme(ct,tp,opts2)
        #dkt_test_models_mcts_qval(ct,tp)
        #dkt_test_models_extract_policy(ct,tp)
        #dkt_test_models_proper_rme(ct,tp,envs)
        dkt_test_models_policy(ct,tp)
    #----------------------------------------------------------------------
    
    ############################################################################
    ############################################################################

    endtime = time.time()
    print('Time elapsed {}s'.format(endtime-starttime))

