# student.py
# @author: Lisa Wang
# @created: Apr 25 2017
#
#===============================================================================
# DESCRIPTION:
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
    def __init__(self, p_trans_satisfied=0.5, p_trans_not_satisfied=0.0, initial_knowledge=0):
        self.p_trans_satisfied = p_trans_satisfied
        self.p_trans_not_satisfied = p_trans_not_satisfied
        if np.sum(initial_knowledge) != 0:
            self.knowledge = initial_knowledge
        else:
            self.knowledge = np.zeros((N_CONCEPTS,))

        # other potential member variables
        # self.motivation = 1


    def do_exercise(self, concept_tree, ex):
        '''
        Simulates solving the provided exercise.
        :param ex: an Exercise object.
        :return: Returns 1 if student solved it correctly, 0 otherwise.
        '''
        # if self._fulfilled_prereqs(ex.concepts):
        if self.fulfilled_prereqs(concept_tree, ex.concepts):
            # print("P trans satisfied_{}".format(self.p_trans_satisfied))
            if np.random.random() <= self.p_trans_satisfied:
                for c in xrange(len(ex.concepts)):
                    if ex.concepts[c] == 1:
                        self.knowledge[c] = 1
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

# END OF class Student
