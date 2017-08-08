# simple_mdp.py
#
#===============================================================================
# DESCRIPTION:
# This module implements a simple MDP model that learns from the student data.
# It treats the students as MDPs and has full access to the student state.
# 
# It also has a simple FMDP model that treats the student as FMDPs.
# It also has full access to the student state
#===============================================================================
# CURRENT STATUS: In Progress
#===============================================================================
# USAGE: import simple_mdp as sm

import numpy as np
import six

from concept_dependency_graph import *
from data_generator import *
from student import Student, Student2
import itertools

class SimpleMDP(object):
    def __init__(self):
        pass
    
    def _k2i(self, knowledge):
        # converts a knowledge numpy array to a state index
        ix = 0
        acc = 1
        for i in xrange(self.n_concepts):
            ix += acc * knowledge[i]
            acc *= 2
        return int(ix)
    
    def _i2k(self, ix):
        # converts a state index to a knowledge numpy array
        knowledge = np.zeros((self.n_concepts,))
        for i in xrange(self.n_concepts):
            knowledge[i] = ix % 2
            ix //= 2
        return knowledge
    
    def _a2i(self, conceptvec):
        # gets the index of a conceptvec
        return np.nonzero(conceptvec)[0][0]
    
    def _reward(self, s, a):
        # a posttest
        return np.sum(self._i2k(s))
    
    def train(self, data):
        '''
        :param data: this is the output from generate_data
        '''
        self.n_concepts = data[0][0][2].shape[0]
        self.n_states = 2**self.n_concepts
        # transition_count[state][action][state] = count
        # visit_count[state][action] = count
        self.visit_count = np.zeros((self.n_states, self.n_concepts), dtype=np.int)
        self.transition_count = np.zeros((self.n_states, self.n_concepts, self.n_states), dtype=np.int)
        # reward will be based on the knowledge state
        
        # now let's go through the data and accumulate the stats
        for i in xrange(len(data)):
            # each trajectory
            # assume students start with concept 0 learned
            curr_s = 1
            for t in xrange(len(data[i])):
                # each timestep and next timestep
                curr_a = self._a2i(data[i][t][0])
                next_s = self._k2i(data[i][t][2])
                self.transition_count[curr_s,curr_a,next_s] += 1
                self.visit_count[curr_s,curr_a] += 1
                # update state tracking
                curr_s = next_s
        
        # make the transition matrix
        self.transition = np.zeros((self.n_states, self.n_concepts, self.n_states))
        # for unvisited state-action pairs, make them self-loops
        for x in xrange(self.n_states):
            for c in xrange(self.n_concepts):
                if self.visit_count[x,c] == 0:
                    # self-loop
                    self.transition[x,c,x] = 1.0
                else:
                    for y in xrange(self.n_states):
                        self.transition[x,c,y] = 1.0 * self.transition_count[x,c,y] / self.visit_count[x,c]
    
    def vi(self, gamma):
        '''
        Do discounted value iteration.
        '''
        # threshold for stopping value iteration
        epsilon = 0.01
        self.q = np.zeros((self.n_states, self.n_concepts))
        maxdiff = 1.0
        while maxdiff > epsilon:
            maxdiff = 0.0
            for x in xrange(self.n_states):
                for c in xrange(self.n_concepts):
                    rhs = 0.0
                    for y in xrange(self.n_states):
                        # compute V(y)
                        next_v = np.max(self.q[y,:])
                        rhs += self.transition[x,c,y]*next_v
                    rhs = self._reward(x,c) + gamma*rhs
                    maxdiff = max(maxdiff, abs(self.q[x,c] - rhs))
                    self.q[x,c] = rhs

class SimpleFMDP(object):
    def __init__(self):
        pass
    
    def _b2i(self, knowledge):
        # converts a binary numpy array to a state index
        ix = 0
        acc = 1
        for i in xrange(self.n_concepts):
            ix += acc * knowledge[i]
            acc *= 2
        return int(ix)
    
    def _i2b(self, ix):
        # converts a state index to a binary numpy array
        state = np.zeros((self.n_concepts,))
        for i in xrange(self.n_concepts):
            state[i] = ix % 2
            ix //= 2
        return state
    
    def _a2i(self, conceptvec):
        # gets the index of a conceptvec
        return np.nonzero(conceptvec)[0][0]
    
    def train(self, data):
        '''
        :param data: this is the output from generate_data
        
        Assume degree is 1.
        '''
        self.n_concepts = data[0][0][2].shape[0]
        self.n_features = data[0][0][3].shape[0]
        self.n_states = 2**self.n_features
        six.print_('{} {} {}'.format(self.n_concepts, self.n_features, self.n_states))
        
        # transition_count[next feature i][action][parent1 feature 1][val1][parent1 feature 2][val2][parent2 feature 2][val2][parent2 feature 2][val2] = count
        # visit_count[action][parent feature 1][val1][parent feature 2][val2][parent2 feature 1][val1][parent2 feature 2][val2] = count
        self.visit_count = np.zeros((self.n_concepts, self.n_features, 2, self.n_features, 2, self.n_features, 2, self.n_features, 2), dtype=np.int)
        self.transition_count = np.zeros((self.n_features, self.n_concepts, self.n_features, 2, self.n_features, 2, self.n_features, 2, self.n_features, 2), dtype=np.int)
        six.print_(self.visit_count.shape)
        six.print_(self.transition_count.shape)
        
        # go through data and collect statistics for p(i | action, j,k) for all features i,j,k
        for i in xrange(len(data)):
            # each trajectory
            # assume students start with concept 0 learned
            for t in xrange(len(data[i])-1):
                # each timestep and next timestep
                curr_s = (data[i][t][3])
                curr_a = self._a2i(data[i][t][0])
                next_s = (data[i][t+1][3])
                #six.print_('{} {} {}'.format(curr_s, curr_a, next_s))
                for (f11,f12,f21,f22) in itertools.product(six.moves.range(self.n_features),repeat=4):
                    self.visit_count[curr_a,f11,curr_s[f11],f12,curr_s[f12],f21,curr_s[f21],f22,curr_s[f22]] += 1
                    for nextf in six.moves.range(self.n_features):
                        self.transition_count[nextf, curr_a,f11,curr_s[f11],f12,curr_s[f12],f21,curr_s[f21],f22,curr_s[f22]] += next_s[nextf]
        
        # for each feature, find its parent
        for f in six.moves.range(self.n_features):
            # try out each feature as a parent, keeping track of which parent has lowest conditional diff
            lowest_diff = None
            best_pf = None
            for (pf1,pf2) in itertools.product(six.moves.range(self.n_features), repeat=2):
                # try 2 parents
                if pf1 == pf2:
                    continue
                worst_diff = None
                for (fv1,fv2) in itertools.product((0,1),repeat=2):
                    # each value of parent set
                    for action in six.moves.range(self.n_concepts):
                        # for each action - we assume all actions have same parent
                        curr_visit_counts = self.visit_count[action,pf1,fv1,pf2,fv2,:,:,:,:]
                        curr_transition_counts = self.transition_count[f,action,pf1,fv1,pf2,fv2,:,:,:,:]
                        curr_count = np.sum(curr_visit_counts)
                        if curr_count <= 0:
                            # didn't see this, so continue
                            continue
                        curr_prob = np.sum(curr_transition_counts) / curr_count
                        # look for the largest diff
                        worst_diff_inner = None
                        for (ppf1,ppf2) in itertools.product(six.moves.range(self.n_features), repeat=2):
                            if ppf1 == ppf2 or ppf1 == pf1 or ppf1 == pf2 or ppf2 == pf1 or ppf2 == pf2:
                                continue
                            for (ffv1,ffv2) in itertools.product((0,1),repeat=2):
                                if curr_visit_counts[ppf1,ffv1,ppf2,ffv2] <= 0:
                                    continue
                                curr_prob_inner = curr_transition_counts[ppf1,ffv1,ppf2,ffv2] / curr_visit_counts[ppf1,ffv1,ppf2,ffv2]
                                curr_diff_inner = np.abs(curr_prob - curr_prob_inner)
                                if worst_diff_inner is None or worst_diff_inner < curr_diff_inner:
                                    worst_diff_inner = curr_diff_inner
                        #six.print_('f {} pf {} fv {} action {} prob {} diffinner {}'.format(f,pf,fv,action,curr_prob,worst_diff_inner))
                        #if worst_diff_inner > 0.5:
                        #    six.print_('{} {}'.format(curr_transition_counts, curr_visit_counts))
                        if worst_diff is None or worst_diff < worst_diff_inner:
                            worst_diff = worst_diff_inner
                if lowest_diff is None or lowest_diff > worst_diff:
                    lowest_diff = worst_diff
                    best_pf = (pf1,pf2)
            six.print_('{} {}'.format(lowest_diff, best_pf))
        pass

def create_custom_dependency():
    '''
    Creates the following dependency tree (where 0 is a prerequisite of 1)
                  0
                 / \
                1   2
               /     \
              3       4
    '''
    dgraph = ConceptDependencyGraph()
    dgraph.n = 5
    dgraph.root = 0
    
    dgraph.children[0] = [1,2]
    dgraph.children[1] = [3]
    dgraph.children[2] = [4]
    
    dgraph.parents[4] = [2]
    dgraph.parents[3] = [1]
    dgraph.parents[2] = [0]
    dgraph.parents[1] = [0]
    
    dgraph._create_prereq_map()
    #print(dgraph.prereq_map)
    
    return dgraph

def expected_reward(data):
    '''
    :param data: output from generate_data
    :return: the sample mean of the posttest reward
    '''
    avg = 0.0
    for i in xrange(len(data)):
        avg += np.mean(data[i][-1][2])
    return avg / len(data)

def expected_sparse_reward(data):
    '''
    :param data: output from generate_data
    :return: the sample mean of the sparse reward
    '''
    avg = 0.0
    for i in xrange(len(data)):
        avg += np.prod(data[i][-1][2])
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

def percent_all_seen(data):
    '''
    Return the percentage of trajectories where all skills other than skill 0 have been tested
    '''
    count = 0.0
    for i in xrange(len(data)):
        seen = data[i][0][0].astype(np.int)
        seen[0] = 1
        for j in xrange(len(data[i])):
            seen += data[i][j][0].astype(np.int)
        if np.all(seen > 0.5):
            count += 1.0
    return count / len(data)

if __name__ == '__main__':
    # test out the model
    n_concepts = 4
    horizon = 5
    
    #dgraph = create_custom_dependency()
    
    dgraph = ConceptDependencyGraph()
    dgraph.init_default_tree(n_concepts)
    
    # custom student
    #student = Student(n=n_concepts,p_trans_satisfied=0.15, p_trans_not_satisfied=0.0, p_get_ex_correct_if_concepts_learned=1.0)
    student2 = Student2(n_concepts, transition_after=True)
    
    data = generate_data(dgraph, student=student2, n_students=10000, filter_mastery=False, seqlen=horizon, policy='random', filename=None, verbose=False)
    print('Average posttest: {}'.format(expected_reward(data)))
    print('Average sprase posttest: {}'.format(expected_sparse_reward(data)))
    print('Percent of full posttest score: {}'.format(percent_complete(data)))
    print('Percent of all seen: {}'.format(percent_all_seen(data)))
    
    fmdp = SimpleFMDP()
    fmdp.train(data)
    