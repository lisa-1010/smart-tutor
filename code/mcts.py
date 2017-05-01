#===============================================================================
# DESCRIPTION:
# An implementation of Monte Carlo Tree Search for POMDPs (TODO ref)
# Takes in a simulator (model) which can generate samples, and maintains a policy.
#===============================================================================

#===============================================================================
# CURRENT STATUS: Needs to be updated with the new Student with separate dependency graph.
#===============================================================================

import numpy as np
import scipy as sp

import copy

import constants
import data_generator as dg
import student as st
import exercise as exer
import dynamics_model_class as dmc

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

class StudentExactSim(object):
    '''
    A model-based simulator for a student. Maintains its own internal hidden state. This wraps around the true simulator.
    '''

    def __init__(self, student, dgraph):
        self.student = student
        self.dgraph = dgraph
    
    def advance_simulator(self, action):
        '''
        Given next action, simulate the student.
        :param action: StudentAction object
        :return: an observation and reward
        '''
        # for now, the reward is the actual student knowledge
        reward = self.student.knowledge[action.concept]
        ob = self.student.do_exercise(self.dgraph, exer.Exercise(action.conceptvec))
        return (ob, reward)
    
    def copy(self):
        '''
        Make a copy of the current simulator.
        '''
        new_knowledge = np.copy(self.student.knowledge)
        new_student = st.Student()
        new_student.knowledge = new_knowledge
        new_copy = StudentExactSim(new_student, self.dgraph)
        return new_copy

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

class StudentExactState(object):
    '''
    The "state" to be used in MCTS. We use the exact student knowledge as the state so this is an MDP
    '''
    def __init__(self, model, sim):
        '''
        :param model: StudentExactSim for the model
        :param sim: StudentExactSim for the real world
        '''
        self.belief = None # not going to use belief at all because we know the exact state
        self.model = model
        self.sim = sim
        self.n_concepts = model.student.knowledge.shape[0]
        
        self.actions = []
        for i in range(self.n_concepts):
            concepts = np.zeros((self.n_concepts,))
            concepts[i] = 1
            self.actions.append(StudentAction(i, concepts))
    
    def perform(self, action):
        # make a copy of the model
        new_model = self.model.copy()
        
        # advance the new model
        new_model.advance_simulator(action)
        
        # create a new state
        new_state = StudentExactState(new_model, self.sim)
        return new_state
    
    def real_world_perform(self, action):
        # first advance the real world simulator
        self.sim.advance_simulator(action)
        
        # make a copy of the real world simulator
        new_model = self.sim.copy()
        
        # use that to create the new state
        new_state = StudentExactState(new_model, self.sim)
        return new_state
    
    def reward(self, parent, action):
        # for now, just use the real knowledge state
        #print('{} {}'.format(self.model.student.knowledge, action.concept))
        return self.model.student.knowledge[action.concept]
    
    def is_terminal(self):
        return False
    
    def __eq__(self, other):
        val = tuple(self.model.student.knowledge)
        oval = tuple(other.model.student.knowledge)
        return val == oval
    
    def __hash__(self):
        # because this is only used for storing a dictionary of immediate children, we can use whatever
        return np.sum(self.model.student.knowledge)
    
    def __str__(self):
        return 'K: {}'.format(self.model.student.knowledge)

def DKTState(object):
    '''
    The belief state to be used in MCTS, implemented using a DKT.
    TODO: needs to be updated when RnnStudentSim is updated
    '''
    def __init__(self, model, sim):
        '''
        :param model: RnnStudentSim object
        :param sim: StudentExactSim object
        '''
        self.belief = model
        # this sim should be shared between all DKTStates
        # and it is advanced only when real_world_perform is called
        # so all references to it will all be advanced
        self.sim = sim
    
    def perform(self, action):
        '''
        Creates a new state where the DKT model is advanced.
        Samples the observation from the DKT model.
        '''
        ob = self.belief.sample_observation(action)
        new_model = self.belief.copy()
        new_model.advance_simulator(action, ob)
        return DKTState(new_model, self.sim)
    
    def real_world_perform(self, action):
        '''
        Advances the true student simulator.
        Creates a new state where the DKT model is advanced according to the result of the true simulator.
        '''
        # advance the true student simulator
        (ob, r) = self.sim.advance_simulator(action)
        new_model = self.belief.copy()
        new_model.advance_simulator(action, ob)
        return DKTState(new_model, self.sim)
    
    def reward(self, parent, action):
        pass
    
    def is_terminal(self):
        pass
    
    def __eq__(self, other):
        # compare the histories
        return self.belief.sequence == other.belief.sequence
    
    def __hash__(self):
        pass
    
    def __str__(self):
        pass


def test_student_sim():
    '''
    TODO there's a bug somewhere right now that prevents the policy from learning
    '''
    import concept_dependency_graph as cdg
    import constants
    horizon = 10
    nrollouts = 50
    n_concepts = 3
    
    random.seed()
    
    concept_tree = cdg.ConceptDependencyGraph()
    concept_tree.init_default_tree(n=n_concepts)
    
    # create the model and simulators
    student = st.Student(p_get_ex_correct_if_concepts_learned=1.0)
    student.knowledge = np.zeros((n_concepts,))
    sim = StudentExactSim(student, concept_tree)
    model = sim.copy()
    
    rollout_policy = default_policies.immediate_reward
    #rollout_policy = default_policies.RandomKStepRollOut(10)
    uct = MCTS(tree_policies.UCB1(1.41), rollout_policy,
               backups.Bellman(0.95))
    
    root = StateNode(None, StudentExactState(model, sim))
    for i in range(horizon):
        print('Step {}'.format(i))
        best_action = uct(root, n=nrollouts)
        print('Current state: {}'.format(str(root.state)))
        print(best_action.concept)
        
        # act in the real environment
        new_root = root.children[best_action].sample_state(real_world=True)
        new_root.parent = None # cutoff the rest of the tree
        root = new_root
        print('Next state: {}'.format(str(new_root.state)))


if __name__ == '__main__':
    test_student_sim()


