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

from simple_mdp import SimpleMDP

from mctslib.graph import *
from mctslib.mcts import *

import mctslib.tree_policies as tree_policies
import mctslib.default_policies as default_policies
import mctslib.backups as backups

# helper functions
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
        # for now, the reward is a full posttest
        reward = np.sum(self.student.knowledge)
        ob = self.student.do_exercise(self.dgraph, exer.Exercise(action.conceptvec))
        return (ob, reward)
    
    def copy(self):
        '''
        Make a copy of the current simulator.
        '''
        new_knowledge = np.copy(self.student.knowledge)
        new_student = copy.copy(self.student)
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
    def __init__(self, model, sim, step, horizon):
        '''
        :param model: StudentExactSim for the model
        :param sim: StudentExactSim for the real world
        :param step: the current timestep (starts from 1)
        :param horizon: the horizon length
        '''
        self.belief = None # not going to use belief at all because we know the exact state
        self.model = model
        self.sim = sim
        self.step = step
        self.horizon = horizon
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
        new_state = StudentExactState(new_model, self.sim, self.step+1, self.horizon)
        return new_state
    
    def real_world_perform(self, action):
        # first advance the real world simulator
        self.sim.advance_simulator(action)
        
        # make a copy of the real world simulator
        new_model = self.sim.copy()
        
        # use that to create the new state
        new_state = StudentExactState(new_model, self.sim, self.step+1, self.horizon)
        return new_state
    
    def reward(self):
        # for now, just use the model knowledge state for a full posttest at the end
        #print('Step {} of {} state {}'.format(self.step, self.horizon, self.model.student.knowledge))
        r = np.sum(self.model.student.knowledge)
        if self.step > self.horizon or True: # toggle between reward every step or only at the end
            return r
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
    
class DKTState(object):
    '''
    The belief state to be used in MCTS, implemented using a DKT.
    '''
    def __init__(self, model, sim, step, horizon, act_hist=[], ob_hist=[]):
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
        
        self.actions = []
        for i in range(self.n_concepts):
            concepts = np.zeros((self.n_concepts,))
            concepts[i] = 1
            self.actions.append(StudentAction(i, concepts))
    
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
        return DKTState(new_model, self.sim, self.step+1, self.horizon, act_hist=new_act_hist, ob_hist=new_ob_hist)
    
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
        return DKTState(new_model, self.sim, self.step+1, self.horizon, act_hist=new_act_hist, ob_hist=new_ob_hist)
    
    def reward(self):
        r = self.belief.sample_reward()
        if self.step > self.horizon or True: # toggle between reward every step or only at the end
            return r
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

def debug_visiter(node, data):
    print('Curr node id: {} n: {} q: {}'.format(id(node), node.n, node.q))
    print('Parent id: {}'.format(str(id(node.parent)) if node.parent is not None else 'None'))
    if isinstance(node, ActionNode):
        print('Action {}'.format(node.action.concept))
    elif isinstance(node, StateNode):
        print('State {} step: {}/{} r: {}'.format(node.state.model.student.knowledge, node.state.step, node.state.horizon, node.reward))
    else:
        print('Not action nor state')

def test_student_exact_single(dgraph, horizon, n_rollouts):
    '''
    Performs a single trajectory with MCTS and returns the final true student knowlegde.
    '''
    n_concepts = dgraph.n
    
    # create the model and simulators
    student = st.Student(p_get_ex_correct_if_concepts_learned=1.0)
    student.knowledge = np.zeros((n_concepts,))
    student.knowledge[0] = 1 # initialize the first concept to be known
    sim = StudentExactSim(student, dgraph)
    model = sim.copy()
    
    #rollout_policy = default_policies.immediate_reward
    rollout_policy = default_policies.RandomKStepRollOut(horizon+1)
    uct = MCTS(tree_policies.UCB1(1.41), rollout_policy,
               backups.monte_carlo)
    
    root = StateNode(None, StudentExactState(model, sim, 1, horizon))
    for i in range(horizon):
        #print('Step {}'.format(i))
        best_action = uct(root, n=n_rollouts)
        #print('Current state: {}'.format(str(root.state)))
        #print(best_action.concept)
        
        # debug check for whether action is optimal
        if True:
            opt_acts = compute_optimal_actions(sim.dgraph, sim.student.knowledge)
            is_opt = best_action.concept in opt_acts
            if not is_opt:
                print('ERROR {} executed non-optimal action {}'.format(sim.student.knowledge, best_action.concept))
                # now let's print out even more debugging information
                #breadth_first_search(root, fnc=debug_visiter)
                #return None
        
        # act in the real environment
        new_root = root.children[best_action].sample_state(real_world=True)
        new_root.parent = None # cutoff the rest of the tree
        root = new_root
        #print('Next state: {}'.format(str(new_root.state)))
    return sim.student.knowledge

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

def test_student_exact():
    '''
    MCTS is now working.
    The number of rollouts required to be optimal grows very fast as a function of the horizon.
    Still, even if not fully optimal, MCTS is an extremely good approximation.
    '''
    import concept_dependency_graph as cdg
    from simple_mdp import create_custom_dependency
    n_concepts = 5
    horizon = 10
    n_rollouts = 50
    n_trajectories = 100
    
    random.seed()
    
    dgraph = create_custom_dependency()
    
    avg = 0.0
    for i in xrange(n_trajectories):
        k = test_student_exact_single(dgraph, horizon, n_rollouts)
        avg += np.mean(k)
    avg = avg / n_trajectories
    
    test_data = dg.generate_data(dgraph, n_students=1000, seqlen=horizon, policy='expert', filename=None, verbose=False)
    print('Average posttest true: {}'.format(expected_reward(test_data)))
    print('Average posttest mcts: {}'.format(avg))


def test_dkt_single(dgraph, horizon, n_rollouts, model):
    '''
    Performs a single trajectory with MCTS and returns the final true student knowlegde.
    '''
    n_concepts = dgraph.n
    
    # create the model and simulators
    student = st.Student(p_get_ex_correct_if_concepts_learned=1.0)
    student.knowledge = np.zeros((n_concepts,))
    student.knowledge[0] = 1 # initialize the first concept to be known
    sim = StudentExactSim(student, dgraph)
    
    # make the model
    model = dmc.RnnStudentSim(model)
    
    #rollout_policy = default_policies.immediate_reward
    rollout_policy = default_policies.RandomKStepRollOut(horizon+1)
    uct = MCTS(tree_policies.UCB1(1.41), rollout_policy,
               backups.monte_carlo)
    
    root = StateNode(None, DKTState(model, sim, 1, horizon))
    for i in range(horizon):
        #print('Step {}'.format(i))
        best_action = uct(root, n=n_rollouts)
        #print('Current state: {}'.format(str(root.state)))
        #print(best_action.concept)
        
        # debug check for whether action is optimal
        if True:
            opt_acts = compute_optimal_actions(sim.dgraph, sim.student.knowledge)
            is_opt = best_action.concept in opt_acts
            if not is_opt:
                print('ERROR {} executed non-optimal action {}'.format(sim.student.knowledge, best_action.concept))
                # now let's print out even more debugging information
                #breadth_first_search(root, fnc=debug_visiter)
                #return None
        
        # act in the real environment
        new_root = root.children[best_action].sample_state(real_world=True)
        new_root.parent = None # cutoff the rest of the tree
        root = new_root
        #print('Next state: {}'.format(str(new_root.state)))
    return sim.student.knowledge

def test_dkt():
    '''
    Test DKT+MCTS
    horizon 5
    Optimal is around 0.69
    With 10000 training samples, DKT gets 0.55
    With 100000 samples DKT gets 0.59
    
    horizon 10
    optimal is around 0.95
    with 100000 is around 0.92
    '''
    import concept_dependency_graph as cdg
    from simple_mdp import create_custom_dependency
    n_concepts = 5
    horizon = 10
    n_rollouts = 50
    n_trajectories = 100
    
    random.seed()
    
    dgraph = create_custom_dependency()
    model_id = 'test_model'
    model = dmc.DynamicsModel(model_id=model_id, timesteps=horizon, load_checkpoint=True)
    
    avg = 0.0
    for i in xrange(n_trajectories):
        k = test_dkt_single(dgraph, horizon, n_rollouts, model)
        avg += np.mean(k)
    avg = avg / n_trajectories
    
    test_data = dg.generate_data(dgraph, n_students=1000, seqlen=horizon, policy='expert', filename=None, verbose=False)
    print('Average posttest true: {}'.format(expected_reward(test_data)))
    print('Average posttest mcts: {}'.format(avg))

if __name__ == '__main__':
    #test_student_exact()
    test_dkt()


