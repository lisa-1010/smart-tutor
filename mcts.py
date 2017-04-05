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

import scikit.mcts as search

class StudentSim(object):
    '''
    A model-based simulator for a student. Maintains its own internal hidden state.
    TODO implement
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
    TODO implement
    '''
    def __init__(self, problem):
        pass
    
    def __eq__(self, other):
        pass

    def __hash__(self):
        pass

class StudentState(object):
    '''
    The "state" to be used in MCTS. It actually represents a history of actions and observations since we are using POMDPs.
    TODO implement
    '''
    def __init__(self, sim):
        pass
    
    def perform(self, action):
        pass
    
    def reward(self, parent, action):
        pass
    
    def is_terminal(self):
        pass
    
    def __eq__(self, other):
        pass
    
    def __hash__(self, other):
        pass

def main():
    pass


if __name__ == '__main__':
    main()


