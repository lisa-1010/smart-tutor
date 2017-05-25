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

import exercise as exer

from mctslib.graph import *
from mctslib.mcts import *

import student as st

from helpers import * # helper functions

# different reward types
DENSE = 0 # posttest every step
SEMISPARSE = 1 # posttest at the end
SPARSE = 2 # product of masteries at the end

class StudentExactState(object):
    '''
    The "state" to be used in MCTS. We use the exact student knowledge as the state so this is an MDP
    '''
    def __init__(self, model, sim, step, horizon, r_type):
        '''
        :param model: StudentExactSim for the model
        :param sim: StudentExactSim for the real world
        :param step: the current timestep (starts from 1)
        :param horizon: the horizon length
        :param r_type: reward type
        '''
        self.belief = None # not going to use belief at all because we know the exact state
        self.model = model
        self.sim = sim
        self.step = step
        self.horizon = horizon
        self.n_concepts = model.student.knowledge.shape[0]
        self.r_type=r_type
        
        self.actions = []
        for i in range(self.n_concepts):
            concepts = np.zeros((self.n_concepts,))
            concepts[i] = 1
            self.actions.append(st.StudentAction(i, concepts))
    
    def perform(self, action):
        # make a copy of the model
        new_model = self.model.copy()
        
        # advance the new model
        new_model.advance_simulator(action)
        
        # create a new state
        new_state = StudentExactState(new_model, self.sim, self.step+1, self.horizon, self.r_type)
        return new_state
    
    def real_world_perform(self, action):
        # first advance the real world simulator
        self.sim.advance_simulator(action)
        
        # make a copy of the real world simulator
        new_model = self.sim.copy()
        
        # use that to create the new state
        new_state = StudentExactState(new_model, self.sim, self.step+1, self.horizon, self.r_type)
        return new_state
    
    def reward(self):
        # for now, just use the model knowledge state for a full posttest at the end
        if self.r_type == DENSE:
            return np.sum(self.model.student.knowledge)
        elif self.step > self.horizon:
            if self.r_type == SEMISPARSE:
                return np.sum(self.model.student.knowledge)
            else:
                # SPARSE
                return np.prod(self.model.student.knowledge)
        else:
            return 0
    
    def is_terminal(self):
        return self.step > self.horizon
    
    def __eq__(self, other):
        val = tuple(self.model.student.knowledge)
        oval = tuple(other.model.student.knowledge)
        return val == oval
    
    def __hash__(self):
        # because this is only used for storing a dictionary of immediate children, we can use whatever
        return k2i(self.model.student.knowledge)
    
    def __str__(self):
        return 'K: {}'.format(self.model.student.knowledge)


class ExactGreedyPolicy(object):
    '''
    Implements a 1-step lookahead with the per-step posttest reward signal for planning with the exact simulator
    '''
    def __init__(self, model, sim):
        '''
        :param model: StudentExactSim object
        :param sim: StudentExactSim object
        '''
        self.model = model
        self.sim = sim
        self.n_concepts = sim.student.knowledge.shape[0]
    
    def best_greedy_action(self, n_rollouts):
        '''
        For each action, samples n_rollouts number of next states and averages the immediate reward.
        Returns the action with the largest next average immediate reward.
        '''
        next_rewards = []
        for a in xrange(self.n_concepts):
            avg_reward = 0.0
            conceptvec = np.zeros((self.n_concepts,))
            conceptvec[a] = 1.0
            action = st.StudentAction(a, conceptvec)
            # sample next state and reward
            for i in xrange(n_rollouts):
                new_model = self.model.copy()
                new_model.advance_simulator(action)
                avg_reward += np.sum(new_model.student.knowledge)
            avg_reward /= 1.0 * n_rollouts
            next_rewards.append(avg_reward)
        #print('{} next {}'.format(self, next_rewards))
        return argmaxlist(next_rewards)[0]
    
    def advance(self, concept):
        '''
        Advances both the simulator and model.
        '''
        conceptvec = np.zeros((self.n_concepts,))
        conceptvec[concept] = 1.0
        action = st.StudentAction(concept, conceptvec)
        
        # first advance the real world simulator
        self.sim.advance_simulator(action)
        
        # make a copy of the real world simulator
        self.model = self.sim.copy()
    
    def __str__(self):
        return 'K {}'.format(self.model.student.knowledge)


class DKTState(object):
    '''
    The belief state to be used in MCTS, implemented using a DKT.
    '''
    def __init__(self, model, sim, step, horizon, r_type, act_hist=[], ob_hist=[]):
        '''
        :param model: RnnStudentSim object
        :param sim: StudentExactSim object
        '''
        # the model will be passed down when doing real world perform
        self.belief = model
        self.step = step
        self.horizon = horizon
        # keep track of history for debugging and various uses
        self.act_hist = act_hist
        self.ob_hist = ob_hist
        # this sim should be shared between all DKTStates
        # and it is advanced only when real_world_perform is called
        # so all references to it will all be advanced
        self.sim = sim
        self.n_concepts = sim.student.knowledge.shape[0]
        self.r_type = r_type
        
        self.actions = []
        for i in range(self.n_concepts):
            concepts = np.zeros((self.n_concepts,))
            concepts[i] = 1
            self.actions.append(st.StudentAction(i, concepts))
    
    def perform(self, action):
        '''
        Creates a new state where the DKT model is advanced.
        Samples the observation from the DKT model.
        '''
        probs = self.belief.sample_observations()
        if probs is None:
            # assume [1 0 0 0 0 ...]
            probs = [0] * self.sim.dgraph.n
            probs[0] = 1
        ob = 1 if np.random.random() < probs[action.concept] else 0
        new_model = self.belief.copy()
        new_model.advance_simulator(action, ob)
        new_act_hist = self.act_hist + [action.concept]
        new_ob_hist = self.ob_hist + [ob]
        return DKTState(new_model, self.sim, self.step+1, self.horizon, self.r_type, act_hist=new_act_hist, ob_hist=new_ob_hist)
    
    def real_world_perform(self, action):
        '''
        Advances the true student simulator.
        Creates a new state where the DKT model is advanced according to the result of the true simulator.
        '''
        # advance the true student simulator
        (ob, r) = self.sim.advance_simulator(action)
        # advance the model with the true observation
        new_model = self.belief.copy()
        new_model.advance_simulator(action, ob)
        new_act_hist = self.act_hist + [action.concept]
        new_ob_hist = self.ob_hist + [ob]
        return DKTState(new_model, self.sim, self.step+1, self.horizon, self.r_type, act_hist=new_act_hist, ob_hist=new_ob_hist)
    
    def reward(self):
        probs = self.belief.sample_observations()
        if probs is None:
            # assume [1 0 0 0 0 ...]
            probs = [0] * self.sim.dgraph.n
            probs[0] = 1
           
        if self.r_type == DENSE:
            return np.sum(probs)
        elif self.step > self.horizon:
            if self.r_type == SEMISPARSE:
                return np.sum(probs)
            else:
                # SPARSE
                return np.prod(probs)
        else:
            return 0
    
    def is_terminal(self):
        return self.step > self.horizon
    
    def __eq__(self, other):
        # compare the histories
        return self.act_hist == other.act_hist and self.ob_hist == other.ob_hist
    
    def __hash__(self):
        # round the predictions first, then convert to index
        probs = self.belief.sample_observations()
        if probs is None:
            # assume [1 0 0 0 0 ...]
            probs = [0] * self.sim.dgraph.n
            probs[0] = 1
        else:
            probs = np.round(probs).astype(np.int)
        return k2i(probs)
    
    def __str__(self):
        probs = self.belief.sample_observations()
        if probs is None:
            # assume [1 0 0 0 0 ...]
            probs = [0] * self.sim.dgraph.n
            probs[0] = 1
        return 'K {} Actions {} Obs {}'.format(probs, self.act_hist, self.ob_hist)


class DKTGreedyPolicy(object):
    '''
    Implements a 1-step lookahead with the per-step posttest reward signal for planning with the DKT
    '''
    def __init__(self, model, sim):
        '''
        :param model: RnnStudentSim object
        :param sim: StudentExactSim object
        '''
        # the model will be passed down when doing real world perform
        self.model = model
        self.sim = sim
        self.n_concepts = sim.student.knowledge.shape[0]
    
    def best_greedy_action(self):
        '''
        For each action, does a 1-step lookahead to determine best action.
        '''
        next_rewards = []
        
        # probability of observations
        probs = self.model.sample_observations()
        if probs is None:
            # assume [1 0 0 0 0 ...]
            probs = [0] * self.sim.dgraph.n
            probs[0] = 1
        
        for a in xrange(self.n_concepts):
            avg_reward = 0.0
            # action
            conceptvec = np.zeros((self.n_concepts,))
            conceptvec[a] = 1.0
            action = st.StudentAction(a, conceptvec)
            # for each observation, weight reward with probability of seeing observation
            new_model = self.model.copy()
            new_model.advance_simulator(action, 1)
            avg_reward += probs[a] * np.sum(new_model.sample_observations())
            new_model = self.model.copy()
            new_model.advance_simulator(action, 0)
            avg_reward += (1.0-probs[a]) * np.sum(new_model.sample_observations())
            # append next reward
            next_rewards.append(avg_reward)
        return argmaxlist(next_rewards)[0]
    
    def advance(self, concept):
        '''
        Advances the true student simulator.
        Creates a new state where the DKT model is advanced according to the result of the true simulator.
        '''
        conceptvec = np.zeros((self.n_concepts,))
        conceptvec[concept] = 1.0
        action = st.StudentAction(concept, conceptvec)
        # advance the true student simulator
        (ob, r) = self.sim.advance_simulator(action)
        # advance the model with the true observation
        self.model.advance_simulator(action, ob)
    
    def __str__(self):
        probs = self.belief.sample_observations()
        if probs is None:
            # assume [1 0 0 0 0 ...]
            probs = [0] * self.sim.dgraph.n
            probs[0] = 1
        return 'K {}'.format(probs)
