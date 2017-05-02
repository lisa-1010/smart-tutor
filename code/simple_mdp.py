# simple_mdp.py
#
#===============================================================================
# DESCRIPTION:
# This module implements a simple MDP model that learns from the student data.
# It treats the students as MDPs and has full access to the student knowledge.
#
#===============================================================================
# CURRENT STATUS: In Progress
#===============================================================================
# USAGE: from simple_mdp import SimpleMDP

import numpy as np

from concept_dependency_graph import *
from data_generator import *

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
        

if __name__ == '__main__':
    # test out the model
    n_concepts = 3
    
    # simple dependency graph
    dgraph = ConceptDependencyGraph()
    dgraph.init_default_tree(n_concepts)
    
    
    data = generate_data(dgraph, n_students=100, seqlen=10, policy='random', filename=None, verbose=False)
    print(data[0])
    
    smdp = SimpleMDP()
    smdp.train(data)
    print(smdp.transition)
    smdp.vi(0.95)
    print(smdp.q)
    