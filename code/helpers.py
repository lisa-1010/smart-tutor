# Helper functions, cna be used for any RL model

import numpy as np
import data_generator as dg


def k2i(knowledge):
    # converts a knowledge numpy array to a state index
    ix = 0
    acc = 1
    for i in xrange(knowledge.shape[0]):
        ix += acc * knowledge[i]
        acc *= 2
    return int(ix)


def i2k(ix, n_concepts):
    # converts a state index to a knowledge numpy array
    knowledge = np.zeros((n_concepts,))
    for i in xrange(n_concepts):
        knowledge[i] = ix % 2
        ix //= 2
    return knowledge


def argmaxlist(xs):
    m = max(xs)
    return [i for i in xrange(len(xs)) if xs[i] == m]


def compute_optimal_actions(concept_tree, knowledge):
    """
    Compute a list of optimal actions (concepts) for the current knowledge.
    """
    opt_acts = []
    for i in xrange(concept_tree.n):
        if not knowledge[i]:
            # if student hasn't learned concept yet:
            # check whether prereqs are fulfilled
            concepts = np.zeros((concept_tree.n,))
            concepts[i] = 1
            if dg.fulfilled_prereqs(concept_tree, knowledge, concepts):
                # this is one optimal action
                opt_acts.append(i)
    if not opt_acts:
        # if no optimal actions, then it means everything is already learned
        # so all actions are optimal
        opt_acts = list(xrange(concept_tree.n))
    return opt_acts


def expected_reward(data):
    '''
    :param data: output from generate_data
    :return: the sample mean of the posttest reward
    '''
    avg = 0.0
    for i in xrange(len(data)):
        avg += np.mean(data[i][-1][2])
    return avg / len(data)


def percent_complete(data):
    '''
    :param data: output from generate_data
    :return: the percentage of trajectories with perfect posttest
    '''
    count = 0.0
    for i in xrange(len(data)):
        if int(np.sum(data[i][-1][2])) == data[i][-1][2].shape[0]:
            count += 1
    return count / len(data)