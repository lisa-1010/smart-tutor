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

import constants
import data_generator as dg
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
            opt_acts = compute_optimal_actions(sim.dgraph, sim.student.knowledge)
            is_opt = best_action.concept in opt_acts  # check if predicted action is optimal

            if not is_opt:
                print('ERROR {} executed non-optimal action {}'.format(sim.student.knowledge, best_action.concept))
                # now let's print out even more debugging information
                #breadth_first_search(root, fnc=debug_visiter)
                #return None

        # act in the real environment
        new_root = root.children[best_action].sample_state(real_world=True) # children of root action nodes, real_world=true advances simulator
        new_root.parent = None # cutoff the rest of the tree
        root = new_root
        #print('Next state: {}'.format(str(new_root.state)))
    return sim.student.knowledge

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


def test_dkt_single(dgraph, s, horizon, n_rollouts, model, r_type, dktcache):
    '''
    Performs a single trajectory with MCTS and returns the final true student knowledge.
    :param dktcache: a dictionary to use for the dkt cache
    '''
    n_concepts = dgraph.n

    # create the model and simulators
    student = s.copy()
    student.reset()
    student.knowledge[0] = 1 # initialize the first concept to be known
    sim = st.StudentExactSim(student, dgraph)

    # make the model
    model = dmc.RnnStudentSim(model)

    #rollout_policy = default_policies.immediate_reward
    rollout_policy = default_policies.RandomKStepRollOut(horizon+1)
    uct = MCTS(tree_policies.UCB1(1.41), rollout_policy,
               backups.monte_carlo) # 1.41 is sqrt (2), backups is from mcts.py

    root = StateNode(None, DKTState(model, sim, 1, horizon, r_type, dktcache))
    for i in range(horizon):
        #print('Step {}'.format(i))
        best_action = uct(root, n=n_rollouts)
        #print('Current state: {}'.format(str(root.state)))
        #print(best_action.concept)

        # debug check for whether action is optimal
        if False:
            opt_acts = compute_optimal_actions(sim.dgraph, sim.student.knowledge)
            is_opt = best_action.concept in opt_acts
            if not is_opt:
                print('ERROR {} executed non-optimal action {}'.format(sim.student.knowledge, best_action.concept))
                # now let's print out even more debugging information
                #breadth_first_search(root, fnc=debug_visiter)
                #return None

        # act in the real environment
        new_root = root.children[best_action].sample_state(real_world=True)
        new_root.parent = None # cutoff the rest of the tree
        root = new_root
        #print('Next state: {}'.format(str(new_root.state)))
    return sim.student.knowledge


def test_dkt_single_greedy(dgraph, s, horizon, model):
    '''
    Performs a single trajectory with greedy 1-step lookahead and returns the final true student knowlegde.
    '''
    n_concepts = dgraph.n

    # create the model and simulators
    student = s.copy()
    student.reset()
    student.knowledge[0] = 1 # initialize the first concept to be known
    sim = st.StudentExactSim(student, dgraph)

    # make the model
    model = dmc.RnnStudentSim(model)

    greedy = DKTGreedyPolicy(model, sim)

    for i in range(horizon):
        best_action = greedy.best_greedy_action()

        # debug check for whether action is optimal
        if False:
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

def test_dkt_chunk(n_trajectories, dgraph, student, model_id, chkpt, horizon, n_rollouts, use_greedy, r_type, dktcache=None):
    '''
    Runs a bunch of trajectories and returns the avg posttest score.
    For parallelization to run in a separate thread/process.
    '''
    # load the model
    if chkpt is not None:
        model = dmc.DynamicsModel(model_id=model_id, timesteps=horizon, load_checkpoint=False)
        model.model.load(chkpt)
    else:
        model = dmc.DynamicsModel(model_id=model_id, timesteps=horizon, load_checkpoint=True)
    # initialize the shared dktcache across MCTS trials
    if dktcache is None:
        dktcache = dict()
    
    acc = 0.0
    for i in xrange(n_trajectories):
        #print('traj i {}'.format(i))
        if not use_greedy:
            k = test_dkt_single(dgraph, student, horizon, n_rollouts, model, r_type, dktcache)
        else:
            k = test_dkt_single_greedy(dgraph, student, horizon, model)
        acc += np.mean(k)
    return acc

def test_dkt(model_id, n_rollouts, n_trajectories, r_type, chkpt=None):
    '''
    Test DKT+MCTS
    '''
    import concept_dependency_graph as cdg
    from simple_mdp import create_custom_dependency
    
    n_concepts = 4
    use_greedy = False
    learn_prob = 0.5
    horizon = 6
    n_jobs = 8
    traj_per_job =  n_trajectories // n_jobs

    #dgraph = create_custom_dependency()

    dgraph = cdg.ConceptDependencyGraph()
    dgraph.init_default_tree(n_concepts)

    #student = st.Student(n=n_concepts,p_trans_satisfied=learn_prob, p_trans_not_satisfied=0.0, p_get_ex_correct_if_concepts_learned=1.0)
    student2 = st.Student2(n_concepts)
    test_student = student2
    
    # create a shared dktcache across all processes
    dktcache_manager = mp.Manager()
    dktcache = dktcache_manager.dict()

    print('Testing model: {}'.format(model_id))
    print('horizon: {}'.format(horizon))
    print('rollouts: {}'.format(n_rollouts))

    accs = Parallel(n_jobs=n_jobs)(delayed(test_dkt_chunk)(traj_per_job, dgraph, test_student, model_id, chkpt, horizon, n_rollouts, use_greedy, r_type, dktcache=dktcache) for _ in range(n_jobs))
    avg = sum(accs) / (n_jobs * traj_per_job)


    test_data = dg.generate_data(dgraph, student=test_student, n_students=1000, seqlen=horizon, policy='expert', filename=None, verbose=False)
    print('Average posttest true: {}'.format(expected_reward(test_data)))
    print('Average posttest mcts: {}'.format(avg))
    return avg

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
    Performs an experiment where every few epochs of training we test with MCTS, and this is repeated.
    '''
    model_id = 'test2_model_small'
    r_type = SEMISPARSE
    dropout = 1.0
    shuffle = False
    n_rollouts = 1000
    n_trajectories = 100
    seqlen = 6
    filename = 'test2-n100000-l{}-random-filtered.pickle'.format(seqlen) # < 6 is already no full mastery

    total_epochs = 12
    epochs_per_iter = 12
    reps = 40
    
    sig_start = mp.Event()
    (p_ch, c_ch) = mp.Pipe()
    training_process = mp.Process(target=staggered_dkt_training,
                                  args=(filename, model_id, seqlen, dropout, shuffle, reps, total_epochs, epochs_per_iter, sig_start, c_ch))
    
    losses = []
    val_losses = []
    score_eps = []
    scores = []
    
    training_process.start()
    for r in xrange(reps):
        losses.append(list())
        val_losses.append(list())
        score_eps.append(list())
        scores.append(list())
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
            scores[r].append(test_dkt(model_id, n_rollouts, n_trajectories, r_type))
    training_process.join() # finish up
    
    np.savez("earlystopping",losses=losses, vals=val_losses, eps=score_eps, scores=scores)


def dkt_pick_best_initialization():
    '''
    Initialize many models and pick based on the training loss.
    Saves the model to a file along with best training loss, so this can be repeated.
    '''
    model_id = 'test2_model_small'
    dropout = 0.7
    shuffle = False
    seqlen = 7 # training data parameter
    reps = 30
    total_epochs = 9 # which epoch's training loss to use
    
    filename = 'test2-n100000-l{}-random-filtered.pickle'.format(seqlen)
    outputstatfile = 'best-loss/{}-dropout{}-epoch{}-{}'.format(model_id, int(dropout * 100), total_epochs, filename)
    modelfile = 'best-loss/{}-dropout{}-epoch{}-{}-checkpoint'.format(model_id, int(dropout * 100), total_epochs, filename)
    print('Output stat file: {}'.format(outputstatfile))
    print('Output model file: {}'.format(modelfile))
    
    # load the saved loss if it exists
    best_loss = None
    try:
        with open(outputstatfile) as f:
            best_loss = pickle.load(f)
    except:
        pass

    losses = []
    
    data = dataset_utils.load_data(filename='{}{}'.format(dg.SYN_DATA_DIR, filename))
    input_data_, output_mask_, target_data_ = dataset_utils.preprocess_data_for_rnn(data)
    train_data = (input_data_[:,:,:], output_mask_[:,:,:], target_data_[:,:,:])
    
    for r in xrange(reps):
        dmodel = dmc.DynamicsModel(model_id=model_id, timesteps=seqlen, dropout=dropout, load_checkpoint=False)
        ecall = ExtractCallback()
        dmodel.train(train_data, n_epoch=total_epochs, callbacks=ecall, shuffle=shuffle, load_checkpoint=False)
        last_loss = ecall.tstates[-1].val_loss # use val loss since dropout messes up train loss
        if best_loss is None or last_loss < best_loss:
            best_loss = last_loss
            # save the model
            dmodel.model.save(modelfile)
        losses.append(last_loss)
    
    # save the best loss
    with open(outputstatfile, 'w') as f:
        pickle.dump(best_loss,f)

    print('Losses Encountered')
    print(losses)
    print('Best loss {}'.format(best_loss))
    
def dkt_train_best_initialization():
    '''
    Load the model saved when picking the best init, and train it some more, leaving the trained model in the normal checkpoints place.
    '''
    model_id = 'test2_model_small'
    
    dropout = 1.0
    shuffle = False
    seqlen = 7 # training data parameter
    filename = 'test2-n100000-l{}-random-filtered.pickle'.format(seqlen)
    modelfile = 'best-loss/{}-dropout{}-shuffle{}-{}-checkpoint'.format(model_id, int(dropout * 100), int(shuffle), filename)
    print('Model file: {}'.format(modelfile))
    
    additional_epochs = 1
    
    data = dataset_utils.load_data(filename='{}{}'.format(dg.SYN_DATA_DIR, filename))
    input_data_, output_mask_, target_data_ = dataset_utils.preprocess_data_for_rnn(data)
    train_data = (input_data_[:,:,:], output_mask_[:,:,:], target_data_[:,:,:])
    

    dmodel = dmc.DynamicsModel(model_id=model_id, timesteps=seqlen, dropout=dropout, load_checkpoint=False)
    # load the model
    dmodel.model.load(modelfile)
    ecall = ExtractCallback()
    dmodel.train(train_data, n_epoch=additional_epochs, callbacks=ecall, shuffle=shuffle, load_checkpoint=False)

if __name__ == '__main__':
    starttime = time.time()

    np.random.seed()
    random.seed()
    
    ######################################
    # testing MCTS and making sure it works when given the true model
    #test_student_exact()
    ######################################
    
    
    ######################################
    # testing early stopping
    test_dkt_early_stopping()
    ######################################
    
    ######################################
    # first pick an initiailization
    #dkt_pick_best_initialization()
    
    # then train the best init
    #dkt_train_best_initialization()
    
    # then test the init
    model_id = 'test2_model_small'
    r_type = SPARSE
    n_rollouts = 300
    n_trajectories = 100
    
    modelfile = 'best-loss/{}-dropout70-epoch9-test2-n100000-l7-random-filtered.pickle-checkpoint'.format(model_id)
    
    #test_dkt(model_id, n_rollouts, n_trajectories, r_type, chkpt=modelfile)
    #######################################

    endtime = time.time()
    print('Time elapsed {}s'.format(endtime-starttime))

