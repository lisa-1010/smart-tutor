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
#           all prequisite concepts have been learned/satisfied
#       - p_trans_not_satisfied: probability of learnign a new concept given
#           that not all prerequisite concepts have been learned
#       - p_get_exercise_correct_if_concepts_learned: probability of getting an
#           exercise correct if all concepts it tests have been learned by the student.
#
#===============================================================================
# CURRENT STATUS: Working
#===============================================================================
# USAGE: from student import *


from __future__ import absolute_import, division, print_function

# Python libraries
import numpy as np
from collections import defaultdict, deque, Counter

# Custom Modules
from constants import *


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
