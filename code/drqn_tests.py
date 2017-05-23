from joblib import Parallel, delayed

import concept_dependency_graph as cdg
from simple_mdp import create_custom_dependency
from helpers import expected_reward, compute_optimal_actions

import student as st
from drqn import *



def test_drqn_single(dgraph, s, horizon, model):
    '''
    Performs a single trajectory with MCTS and returns the final true student knowledge.
    '''
    n_concepts = dgraph.n

    # create the model and simulators
    student = s.copy()
    student.reset()
    student.knowledge[0] = 1  # initialize the first concept to be known
    sim = st.StudentExactSim(student, dgraph)

    # TODO: create start state

    for i in range(horizon):
        # print('Step {}'.format(i))

        best_action = model.predict()
        # print('Current state: {}'.format(str(root.state)))
        # print(best_action.concept)

        # debug check for whether action is optimal
        if False:
            opt_acts = compute_optimal_actions(sim.dgraph, sim.student.knowledge) # put function code into shared file
            is_opt = best_action.concept in opt_acts
            if not is_opt:
                print('ERROR {} executed non-optimal action {}'.format(sim.student.knowledge,
                                                                       best_action.concept))
                # now let's print out even more debugging information
                # breadth_first_search(root, fnc=debug_visiter)
                # return None

        # act in the real environment
        new_root = root.children[best_action].sample_state(real_world=True)
        new_root.parent = None  # cutoff the rest of the tree
        root = new_root
        # print('Next state: {}'.format(str(new_root.state)))
    return sim.student.knowledge


def test_drqn_chunk(n_trajectories, dgraph, student, model_id, horizon):
    '''
    Runs a bunch of trajectories and returns the avg posttest score.
    For parallelization to run in a separate thread/process.
    '''
    model = DRQNModel(model_id=model_id, timesteps=horizon, load_checkpoint=True)

    acc = 0.0
    for i in xrange(n_trajectories):
        print('traj i {}'.format(i))

        k = test_drqn_single(dgraph, student, horizon, model)

        acc += np.mean(k)
    return acc


def test_drqn():
    '''
    Test DRQN
    '''
    n_concepts = 4
    learn_prob = 0.15
    horizon = 6
    n_rollouts = 200
    n_trajectories = 100
    n_jobs = 8
    traj_per_job = n_trajectories // n_jobs

    dgraph = create_custom_dependency()

    # dgraph = cdg.ConceptDependencyGraph()
    # dgraph.init_default_tree(n_concepts)

    student = st.Student(n=n_concepts, p_trans_satisfied=learn_prob, p_trans_not_satisfied=0.0, p_get_ex_correct_if_concepts_learned=1.0)

    model_id = 'test2_model_mid'

    print('Testing model: {}'.format(model_id))
    print('horizon: {}'.format(horizon))
    print('rollouts: {}'.format(n_rollouts))

    accs = Parallel(n_jobs=n_jobs)(delayed(test_drqn_chunk)(traj_per_job, dgraph, student, model_id, horizon) for _ in range(n_jobs))
    avg = sum(accs) / (n_jobs * traj_per_job)

    test_data = dg.generate_data(dgraph, student=student, n_students=1000, seqlen=horizon, policy='expert', filename=None, verbose=False)
    print('Average posttest true: {}'.format(expected_reward(test_data)))
    print('Average posttest drqn: {}'.format(avg))
