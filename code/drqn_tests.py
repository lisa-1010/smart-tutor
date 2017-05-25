from joblib import Parallel, delayed

import concept_dependency_graph as cdg
from simple_mdp import create_custom_dependency
from helpers import expected_reward, compute_optimal_actions

import student as st
from drqn import *

def construct_drqn_inputs(act_hist, ob_hist, n_concepts):
    n_timesteps = len(act_hist)
    inputs = np.zeros((1, n_timesteps, 2 * n_concepts))
    for t in xrange(n_timesteps):
        action, ob = act_hist[t], ob_hist[t]
        exer = np.zeros(n_concepts)
        exer[action] = 1
        if ob == 1:
            inputs[0, t, :n_concepts] = exer
        else:
            inputs[0, t, n_concepts:] = exer
    return inputs



def test_drqn_single(dgraph, student, horizon, model, DEBUG=False):
    '''
    Performs a single trajectory with MCTS and returns the final true student knowledge.
    '''
    n_concepts = dgraph.n

    # create the model and simulators
    student.reset()
    student.knowledge[0] = 1  # initialize the first concept to be known
    sim = st.StudentExactSim(student, dgraph)

    # initialize state (or alternatively choose random first action)
    act_hist = [0]
    ob_hist = [0]
    for i in range(horizon - 1):
        # print('Step {}'.format(i))
        inputs = construct_drqn_inputs(act_hist, ob_hist, n_concepts)
        best_action, _ = model.predict(inputs, last_timestep_only=True)
        best_action = best_action[0]
        concept = best_action
        conceptvec = np.zeros(n_concepts)
        conceptvec[concept] = 1
        action = st.StudentAction(concept, conceptvec)
        # print(best_action.concept)

        # debug check for whether action is optimal
        if DEBUG:
            opt_acts = compute_optimal_actions(sim.dgraph, sim.student.knowledge) # put function code into shared file
            is_opt = action.concept in opt_acts
            if not is_opt:
                print('ERROR {} executed non-optimal action {}'.format(sim.student.knowledge,
                                                                       action.concept))

        # act in the real environment
        (ob, reward) = sim.advance_simulator(action)
        act_hist.append(action.concept)
        ob_hist.append(ob)

        # print('Next state: {}'.format(str(new_root.state)))
    return sim.student.knowledge


def test_drqn_chunk(n_trajectories, dgraph, student, model_id, horizon):
    '''
    Runs a bunch of trajectories and returns the avg posttest score.
    For parallelization to run in a separate thread/process.
    '''

    model = DRQNModel(model_id=model_id, timesteps=horizon)
    model.init_evaluator()

    acc = 0.0
    for i in xrange(n_trajectories):
        print('traj i {}'.format(i))
        k = test_drqn_single(dgraph, student, horizon, model)
        acc += np.mean(k)

    acc /= n_trajectories
    return acc


def test_drqn(model_id="", parallel=False):
    '''
    Test DRQN
    '''
    n_concepts = 4
    learn_prob = 0.15
    horizon = 6
    n_trajectories = 100
    n_jobs = 8
    traj_per_job = n_trajectories // n_jobs

    from simple_mdp import create_custom_dependency
    # dgraph = create_custom_dependency()

    dgraph = cdg.ConceptDependencyGraph()
    dgraph.init_default_tree(n_concepts)

    # student = st.Student(n=n_concepts, p_trans_satisfied=learn_prob, p_trans_not_satisfied=0.0, p_get_ex_correct_if_concepts_learned=1.0)
    student = st.Student2(n_concepts)
    if model_id == "":
        model_id = "test_model_drqn"

    print('Testing model: {}'.format(model_id))
    print('horizon: {}'.format(horizon))

    if parallel:
        accs = Parallel(n_jobs=n_jobs)(delayed(test_drqn_chunk)(traj_per_job, dgraph, student, model_id, horizon) for _ in range(n_jobs))
        avg = sum(accs) / (n_jobs)
    else:
        avg = test_drqn_chunk(n_trajectories, dgraph, student, model_id, horizon)

    test_data = dg.generate_data(dgraph, student=student, n_students=1000, seqlen=horizon, policy='expert', filename=None, verbose=False)
    print('Average posttest true: {}'.format(expected_reward(test_data)))
    print('Average posttest drqn: {}'.format(avg))
