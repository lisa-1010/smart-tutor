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
# USAGE: import simple_mdp as sm

import numpy as np

from concept_dependency_graph import *
from data_generator import *
from student import Student

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
    horizon = 7
    
    #dgraph = create_custom_dependency()
    
    dgraph = ConceptDependencyGraph()
    dgraph.init_default_tree(n_concepts)
    
    # custom student
    #student = Student(n=n_concepts,p_trans_satisfied=0.15, p_trans_not_satisfied=0.0, p_get_ex_correct_if_concepts_learned=1.0)
    student2 = Student2(n_concepts)
    
    data = generate_data(dgraph, student=student2, n_students=100000, seqlen=horizon, policy='random', filename=None, verbose=False)
    print('Average posttest: {}'.format(expected_reward(data)))
    print('Percent of full posttest score: {}'.format(percent_complete(data)))
    print('Percent of all seen: {}'.format(percent_all_seen(data)))
    # for seqlen=5 expert=0.68
    # seqlen=3 seems to be the point where there should be enough information to generalize the optimal policy
    
    #smdp = SimpleMDP()
    #smdp.train(data)
    #smdp.vi(0.95)
    '''
    What should be the optimal policy?
    00000
    10000 [1 2]
    01000
    11000 [2 3]
    00100
    10100 [1 4]
    01100
    11100 [3 4]
    00010
    10010
    01010
    11010 [2]
    00110
    10110
    01110
    11110 [4]
    00001
    10001
    01001
    11001
    00101
    10101 [1]
    01101
    11101 [3]
    00011
    10011
    01011
    11011
    00111
    10111
    01111
    11111
    The states left blank are impossible states
    '''
    #for i,acts in [(1,[1,2]),(3,[2,3]),(5,[1,4]),(7,[3,4]),(11,[2]),(15,[4]),(21,[1]),(23,[3])]:
    #    policy_a = np.argmax(smdp.q[i,:])
    #    print('{} {}'.format(policy_a in acts, smdp.q[i,:]))
    