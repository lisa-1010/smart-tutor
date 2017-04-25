# exercise.py
# @author: Lisa Wang
# @created: Apr 25 2017
#
#===============================================================================
# DESCRIPTION:
#
#===============================================================================
# CURRENT STATUS: Working
#===============================================================================
# USAGE: from exercise import *


from __future__ import absolute_import, division, print_function

# Python libraries
import numpy as np
from collections import defaultdict, deque, Counter

# Custom Modules
from constants import *


class Exercise(object):
    def __init__(self, concepts=0):
        '''
        :param concepts: a binary np.array encoding the concepts practiced by this exercise.
        Could be one-hot for simple model, so each exercise practices exactly one concept.
        '''
        # if concepts is None, a random concept is chosen.
        if np.sum(concepts) != 0:
            self.concepts = concepts
        else:
            # create a one hot vector for concepts
            self.concepts = np.zeros((N_CONCEPTS,))
            self.concepts[random.randint(0, N_CONCEPTS - 1)] = 1
# End of class Exercise
