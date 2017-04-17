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

import data_generator as dg

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
    def __init__(self, concept):
        self.concept = concept
    
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
    
    def perform(self, action):
        ex = dg.Exercise(concept=action.concept)
        old_knowledge = self.belief[0].knowledge
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
        ex = dg.Exercise(concept=action.concept)
        old_knowledge = self.belief[0].knowledge
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
    
    def reward(self, parent, action):
        pass
    
    def is_terminal(self):
        return False
    
    def __eq__(self, other):
        val = (self.belief[0].knowledge, self.belief[1], self.belief[2])
        oval = (other.belief[0].knowledge, other.belief[1], other.belief[2])
        return val == oval
    
    def __hash__(self):
        # take a shortcut and only compare last concept and last observation
        # because this is only used for storing a dictionary of immediate children (double check this)
        return new_state.belief[1][-1]*10 + new_state.belief[2]

def main():
    pass


if __name__ == '__main__':
    main()


