#===============================================================================
# DESCRIPTION:
# Tests on DKT + MCTS
#===============================================================================

#===============================================================================
# CURRENT STATUS:
#===============================================================================

import numpy as np
import scipy as sp

import time
import copy

import constants
import data_generator as dg
import student as st
import exercise as exer
import dynamics_model_class as dmc

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
    StudentExactState



def debug_visiter(node, data):
    print('Curr node id: {} n: {} q: {}'.format(id(node), node.n, node.q))
    print('Parent id: {}'.format(str(id(node.parent)) if node.parent is not None else 'None'))
    if isinstance(node, ActionNode):
        print('Action {}'.format(node.action.concept))
    elif isinstance(node, StateNode):
        print('State {} step: {}/{} r: {}'.format(node.state.model.student.knowledge, node.state.step, node.state.horizon, node.reward))
    else:
        print('Not action nor state')



def test_student_exact_single(dgraph, stud, horizon, n_rollouts):
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

    root = StateNode(None, StudentExactState(model, sim, 1, horizon)) # create root node of the tree, 1 is the step.
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


def test_student_exact_chunk(n_trajectories, dgraph, student, horizon, n_rollouts, use_greedy):
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
            k = test_student_exact_single(dgraph, student, horizon, n_rollouts)
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
    n_concepts = 4
    learn_prob = 0.5
    horizon = 7
    n_rollouts = 50
    n_trajectories = 100
    n_jobs = 8
    traj_per_job =  n_trajectories // n_jobs

    #dgraph = create_custom_dependency()

    dgraph = cdg.ConceptDependencyGraph()
    dgraph.init_default_tree(n_concepts)

    student = st.Student(n=n_concepts,p_trans_satisfied=learn_prob, p_trans_not_satisfied=0.0, p_get_ex_correct_if_concepts_learned=1.0)
    student2 = st.Student2(n_concepts)
    test_student = student2

    accs = Parallel(n_jobs=n_jobs)(delayed(test_student_exact_chunk)(traj_per_job, dgraph, test_student, horizon, n_rollouts, use_greedy) for _ in range(n_jobs))
    avg = sum(accs) / (n_jobs * traj_per_job)

    test_data = dg.generate_data(dgraph, student=test_student, n_students=1000, seqlen=horizon, policy='expert', filename=None, verbose=False)
    print('Number of jobs {}'.format(n_jobs))
    print('Trajectory per job {}'.format(traj_per_job))
    print('Use greedy? {}'.format(use_greedy))
    print('Average posttest true: {}'.format(expected_reward(test_data)))
    print('Average posttest mcts: {}'.format(avg))


def test_dkt_single(dgraph, s, horizon, n_rollouts, model):
    '''
    Performs a single trajectory with MCTS and returns the final true student knowledge.
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

    root = StateNode(None, DKTState(model, sim, 1, horizon))
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

def test_dkt_chunk(n_trajectories, dgraph, student, model_id, horizon, n_rollouts, use_greedy):
    '''
    Runs a bunch of trajectories and returns the avg posttest score.
    For parallelization to run in a separate thread/process.
    '''
    model = dmc.DynamicsModel(model_id=model_id, timesteps=horizon, load_checkpoint=True)
    acc = 0.0
    for i in xrange(n_trajectories):
        print('traj i {}'.format(i))
        if not use_greedy:
            k = test_dkt_single(dgraph, student, horizon, n_rollouts, model)
        else:
            k = test_dkt_single_greedy(dgraph, student, horizon, model)
        acc += np.mean(k)
    return acc

def test_dkt():
    '''
    Test DKT+MCTS
    '''
    import concept_dependency_graph as cdg
    from simple_mdp import create_custom_dependency
    n_concepts = 4
    use_greedy = False
    learn_prob = 0.15
    horizon = 6
    n_rollouts = 200
    n_trajectories = 100
    n_jobs = 8
    traj_per_job =  n_trajectories // n_jobs

    #dgraph = create_custom_dependency()

    dgraph = cdg.ConceptDependencyGraph()
    dgraph.init_default_tree(n_concepts)

    #student = st.Student(n=n_concepts,p_trans_satisfied=learn_prob, p_trans_not_satisfied=0.0, p_get_ex_correct_if_concepts_learned=1.0)
    student2 = st.Student2(n_concepts)
    test_student = student2

    #model_id = 'test_model_small'
    #model_id = 'test_model_mid'
    #model_id = 'test_model'

    model_id = 'test2_model_mid'

    print('Testing model: {}'.format(model_id))
    print('horizon: {}'.format(horizon))
    print('rollouts: {}'.format(n_rollouts))

    accs = Parallel(n_jobs=n_jobs)(delayed(test_dkt_chunk)(traj_per_job, dgraph, test_student, model_id, horizon, n_rollouts, use_greedy) for _ in range(n_jobs))
    avg = sum(accs) / (n_jobs * traj_per_job)


    test_data = dg.generate_data(dgraph, student=test_student, n_students=1000, seqlen=horizon, policy='expert', filename=None, verbose=False)
    print('Average posttest true: {}'.format(expected_reward(test_data)))
    print('Average posttest mcts: {}'.format(avg))

if __name__ == '__main__':
    starttime = time.time()

    np.random.seed()
    random.seed()


    #test_student_exact()
    test_dkt()

    endtime = time.time()
    print('Time elapsed {}s'.format(endtime-starttime))

