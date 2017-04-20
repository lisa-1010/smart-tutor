#===============================================================================
# DESCRIPTION:
# An implementation of Monte Carlo Tree Search for POMDPs (TODO ref)
# Takes in a simulator (model) which can generate samples, and maintains a policy.
#===============================================================================

#===============================================================================
# CURRENT STATUS: Barely starting
#===============================================================================

import numpy as np
import scipy as sp

import constants
import data_generator as dg

from mctslib.graph import StateNode
from mctslib.mcts import *

import mctslib.tree_policies as tree_policies
import mctslib.default_policies as default_policies
import mctslib.backups as backups

class StudentSim(object):
    '''
    A model-based simulator for a student. Maintains its own internal hidden state.
    This is just a template.
    '''

    def __init__(self, data):
        pass
    
    def sample_observation(self, action):
        '''
        Samples a new observation given an action.
        '''
        pass
    
    def sample_reward(self, action):
        '''
        Samples a new reward given an action.
        '''
        pass
    
    def advance_simulator(self, action, observation):
        '''
        Given next action and observation, advance the internal hidden state of the simulator.
        '''
        pass
    
    def copy(self):
        '''
        Make a copy of the current simulator.
        '''
        pass

class StudentAction(object):
    '''
    Represents an action of the tutor in MCTS i.e. a problem to give to the student.
    '''
    def __init__(self, concept, conceptvec):
        self.concept = concept
        self.conceptvec = conceptvec
    
    def __eq__(self, other):
        return self.concept == other.concept

    def __hash__(self):
        return self.concept

class StudentState(object):
    '''
    The "state" to be used in MCTS. It actually represents a history of actions and observations since we are using POMDPs.
    TODO implement
    '''
    def __init__(self, student):
        # student, history of exercises, history of obs
        self.belief = [student, [], []]
        self.actions = []
        for i in range(dg.N_CONCEPTS):
            concepts = np.zeros((dg.N_CONCEPTS,))
            concepts[i] = 1
            self.actions.append(StudentAction(i, concepts))
    
    def perform(self, action):
        # create exercise
        concepts = np.zeros((dg.N_CONCEPTS,))
        concepts[action.concept] = 1
        ex = dg.Exercise(concepts=concepts)
        
        # advance the simulator
        old_knowledge = np.copy(self.belief[0].knowledge)
        result = self.belief[0].do_exercise(ex)
        new_knowledge = self.belief[0].knowledge
        self.belief[0].knowledge = old_knowledge # revert back current sim
        
        # update history for the new state
        new_exes = self.belief[1] + [action.concept]
        new_obs = self.belief[2] + [result]
        
        # create and update knowledge of simulator and history of new state
        new_student = dg.Student()
        new_student.knowledge = new_knowledge
        new_state = StudentState(new_student)
        new_state.belief[1] = new_exes
        new_state.belief[2] = new_obs
        
        return new_state
    
    def real_world_perform(self, action):
        return self.perform(action)
    
    def reward(self, parent, action):
        # for now, just use the real knowledge state
        print('{} {}'.format(self.belief[0].knowledge, action.concept))
        return np.sum(self.belief[0].knowledge)
    
    def is_terminal(self):
        return False
    
    def __eq__(self, other):
        val = (list(self.belief[0].knowledge), self.belief[1], self.belief[2])
        oval = (list(other.belief[0].knowledge), other.belief[1], other.belief[2])
        return val == oval
    
    def __hash__(self):
        # take a shortcut and only compare last concept and last observation
        # because this is only used for storing a dictionary of immediate children (double check this)
        return int(self.belief[1][-1]*10 + self.belief[2][-1])
    
    def __str__(self):
        return 'EX: {} C: {} K: {}'.format(self.belief[1],self.belief[2],self.belief[0].knowledge)

        
def test_student_sim():
    horizon = 4
    nrollouts = 50
    
    random.seed()
    
    rollout_policy = default_policies.immediate_reward
    #rollout_policy = default_policies.RandomKStepRollOut(10)
    uct = MCTS(tree_policies.UCB1(1.41), rollout_policy,
               backups.Bellman(0.95))
    
    root = StateNode(None, StudentState(dg.Student()))
    for i in range(horizon):
        print('Step {}'.format(i))
        best_action = uct(root, n=nrollouts)
        print('Current state: {}'.format(str(root.state)))
        #print(best_action)
        
        # act in the real environment
        new_root = root.children[best_action].sample_state(real_world=True)
        new_root.parent = None # cutoff the rest of the tree
        root = new_root
        print('Next state: {}'.format(str(new_root.state)))


if __name__ == '__main__':
    test_student_sim()


