# student.py
# @author: Lisa Wang
# @created: Apr 25 2017
#
#===============================================================================
# DESCRIPTION:
# This module defines the student model used by the data generator / simulator.
# This object keeps track of a student's knowledge (latent state)
# The student model is defined by three probabilities:
#       - p_trans_satisfied: probability of learning a new concept given that
#           all prerequisite concepts have been learned/satisfied
#       - p_trans_not_satisfied: probability of learning a new concept given
#           that not all prerequisite concepts have been learned
#       - p_get_exercise_correct_if_concepts_learned: probability of getting an
#           exercise correct if all concepts it tests have been learned by the student.
#
#===============================================================================
# CURRENT STATUS: Working
#===============================================================================
# USAGE: import student as st


from __future__ import absolute_import, division, print_function

# Python libraries
import numpy as np
from collections import defaultdict, deque, Counter

# Custom Modules
from constants import *
import exercise as exer

import dynamics_model_class as dmc


class Student(object):
    def __init__(self, n=None, p_trans_satisfied=0.5, p_trans_not_satisfied=0.0, p_get_ex_correct_if_concepts_learned=1.0, initial_knowledge=0):
        self.p_trans_satisfied = p_trans_satisfied
        self.p_trans_not_satisfied = p_trans_not_satisfied
        self.p_get_ex_correct_if_concepts_learned = p_get_ex_correct_if_concepts_learned
        n_concepts = n if n is not None else N_CONCEPTS
        if np.sum(initial_knowledge) != 0:
            self.knowledge = initial_knowledge
        else:
            self.knowledge = np.zeros((n_concepts,))
    
    def reset(self):
        '''
        Reset to initial condition so that we can start simulating from the beginning again.
        '''
        self.knowledge = np.zeros(self.knowledge.shape[0])
    
    def copy(self):
        '''
        Copies this generator.
        '''
        new_student = Student()
        new_student.p_trans_satisfied = self.p_trans_satisfied
        new_student.p_trans_not_satisfied = self.p_trans_not_satisfied
        new_student.p_get_ex_correct_if_concepts_learned = self.p_get_ex_correct_if_concepts_learned
        new_student.knowledge = np.copy(self.knowledge)
        return new_student

    def do_exercise(self, concept_tree, ex):
        '''
        Simulates solving the provided exercise.
        :param ex: an Exercise object.
        :return: Returns 1 if student solved it correctly, 0 otherwise.
        '''
        # if self._fulfilled_prereqs(ex.concepts):
        if self.fulfilled_prereqs(concept_tree, ex.concepts):
            # print("P trans satisfied_{}".format(self.p_trans_satisfied))
            for c in xrange(len(ex.concepts)):
                if ex.concepts[c] == 1 and np.random.random() <= self.p_trans_satisfied:
                    # update latent knowledge state
                    self.knowledge[c] = 1
            if self.learned_all_concepts_in_ex(ex.concepts) and np.random.random() <= self.p_get_ex_correct_if_concepts_learned:
                return 1
            else:
                return 0
        else:
            return 1 if np.random.random() <= self.p_trans_not_satisfied else 0


    def fulfilled_prereqs(self, concept_tree, concepts):
        '''
        for each concept tested in the exercise, check if all prereqs are fulfilled.
        if prereqs for at least one concept are not fulfilled, then function returns False.
        :return: bool
        '''
        for i in xrange(len(concepts)):
            c = concepts[i]
            if c == 1:
                prereqs = concept_tree.get_prereqs(i)
                if np.sum(np.multiply(self.knowledge, prereqs)) != np.sum(prereqs):
                    return False
        return True

    def learned_all_concepts_in_ex(self, concepts):
        for c in xrange(len(concepts)):
            if concepts[c] == 1 and self.knowledge[c] == 0:
                return False
        return True

    # END OF class Student


class Student2(object):
    '''
    Special Deterministic Student to facilitate testing. Should be easier to learn than probabilistic student.
    Instead of a probability of mastering a skill when prereqs are fulfilled, always need exactly two tries.
    This means the first try is always a fail. And only after the second try does the student learn.
    Deterministic observations still.
    
    Can either transition before the observation or after the observation.
    This means either the second try is always a pass, or it's a fail and the 3rd try is a pass.
    '''
    def __init__(self, n_concepts, transition_after):
        self.knowledge = np.zeros((n_concepts,))
        self.visited = np.zeros((n_concepts,)).astype(np.int)
        self.transition_after = transition_after

    def reset(self):
        self.knowledge = np.zeros(self.knowledge.shape)
        self.visited = np.zeros(self.knowledge.shape).astype(np.int)

    def copy(self):
        '''
        Copies this generator.
        '''
        new_student = Student2(1, False)
        new_student.knowledge = np.copy(self.knowledge)
        new_student.visited = np.copy(self.visited)
        new_student.transition_after = self.transition_after
        return new_student
    
    def update_knowledge(self, concept_tree, ex):
        '''
        Half of an update. This updates the student's knowledge.
        This can be called before, or after the observation.
        '''
        if self.fulfilled_prereqs(concept_tree, ex.concepts):
            for c in xrange(len(ex.concepts)):
                if ex.concepts[c] == 1:
                    # has been visited before?
                    if self.visited[c] >= 1:
                        # if yes, then this is second time visited so yes mastery
                        self.knowledge[c] = 1
                    # concept has been visited
                    self.visited[c] = 1
    
    def try_exercise(self, concept_tree, ex):
        '''
        Get an observation of whether the student gets the question correct without updating the knowledge.
        '''
        return self.learned_all_concepts_in_ex(ex.concepts)
    
    def do_exercise(self, concept_tree, ex):
        '''
        Simulates solving the provided exercise.
        :param ex: an Exercise object.
        :return: Returns 1 if student solved it correctly, 0 otherwise.
        '''
        if not self.transition_after:
            self.update_knowledge(concept_tree, ex)
            ob = self.try_exercise(concept_tree, ex)
        else:
            ob = self.try_exercise(concept_tree, ex)
            self.update_knowledge(concept_tree, ex)
        return ob


    def fulfilled_prereqs(self, concept_tree, concepts):
        '''
        for each concept tested in the exercise, check if all prereqs are fulfilled.
        if prereqs for at least one concept are not fulfilled, then function returns False.
        :return: bool
        '''
        for i in xrange(len(concepts)):
            c = concepts[i]
            if c == 1:
                prereqs = concept_tree.get_prereqs(i)
                if np.sum(np.multiply(self.knowledge, prereqs)) != np.sum(prereqs):
                    return False
        return True

    def learned_all_concepts_in_ex(self, concepts):
        for c in xrange(len(concepts)):
            if concepts[c] == 1 and self.knowledge[c] == 0:
                return False
        return True

    # END OF class Student2


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
    
    def get_knowledge(self):
        return self.student.knowledge

    def copy(self):
        '''
        Make a copy of the current simulator.
        '''
        new_student = self.student.copy()
        new_copy = StudentExactSim(new_student, self.dgraph)
        return new_copy


class StudentAction(object):
    '''
    Represents an action of the tutor, i.e. a problem to give to the student.
    Purpose: To facilitate swapping between vectorized and indexed representation of concepts.
    '''
    def __init__(self, concept, conceptvec):
        self.concept = concept
        self.conceptvec = conceptvec

    def __eq__(self, other):
        return self.concept == other.concept

    def __hash__(self):
        return self.concept

class StudentDKTSim(object):
    '''
    A model-based simulator for a student. Maintains its own internal history. This wraps around a DKT, which is maintained in a separate process in order to not conflict with stuff in the current thread. Also uses a cache to help speed things up.
    '''

    def __init__(self, dgraph, dmcmodel, dktcache, histhash=''):
        '''
        Wraps around a given model (could be a proxy from a Manager or not)
        '''
        self.dgraph = dgraph
        self.dkt = dmc.RnnStudentSim(dmcmodel)
        self.dktcache = dktcache
        self.histhash = histhash
    
    def _next_histhash(self, new_act, new_ob):
        return self.histhash + '{}{};'.format(new_act, new_ob)

    def get_probs(self):
        # computes and caches the probs for the current state
        # try the dktcache
        trycache = self.dktcache.get(self.histhash, None)

        if trycache is None:
            # actually run it and update the cache
            trycache = self.dkt.sample_observations()
            if trycache is None:
                trycache = np.array([0.0] * self.dgraph.n)
                trycache[0] = 1.0
                # cache back
                self.dktcache[self.histhash] = trycache
        return trycache
    
    def get_knowledge(self):
        return self.get_probs()
    
    def advance_simulator(self, action):
        '''
        Given next action, simulate the student and advance the simulator.
        :param action: StudentAction object
        :return: an observation and reward
        '''
        probs = self.get_probs()
        # for now, the reward is a full posttest
        reward = np.sum(probs)
        ob = 1 if np.random.random() < probs[action.concept] else 0
        # advance the simulator
        self.histhash = self._next_histhash(action.concept,ob)
        self.dkt.advance_simulator(action,ob)
        return (ob, reward)

    def copy(self):
        '''
        Make a copy of the current simulator.
        '''
        new_copy = StudentDKTSim(self.dgraph, self.dkt.model, self.dktcache, self.histhash)
        new_copy.dkt = self.dkt.copy()
        return new_copy