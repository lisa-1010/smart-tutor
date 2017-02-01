# student_model.py
# @author: Lisa Wang
# @created: Jan 30 2016
#
#===============================================================================
# DESCRIPTION:
# Ground truth student model for data generation.
#
#===============================================================================
# CURRENT STATUS: In progress
#===============================================================================
# USAGE:

import numpy as np
import random


# Params / Constants
N_CONCEPTS = 10
N_EXERCISES = 100


class ConceptDependencyTree(object):
    def __init__(self):
        self.root = None
        self.edges = [] # edges go from parent (e.g. prerequisite) to child


    def init_default_tree(self, n):
        '''
        Creates a balanced binary tree (Where A - H are concepts)
        and B depends on A, etc.
                    A
                 /     \
                B       C
               / \     / \
              E  F    G   H

        :param n:
        :return:
        '''
        # n: number of nodes
        assert (n > 0), "Tree must have at least one node."
        self.root = 0
        for i in xrange(0, n):
            if 2 * i + 1 < n:
                self.edges.append((i, 2 * i + 1))
            else:
                # for leaf nodes, add a pseudo edge pointing to -1.
                self.edges.append((i, -1))
            if 2 * i + 2 < n:
                self.edges.append((i, 2 * i + 2))

    def print_edges(self):
        print self.edges



class Exercise(object):
    def __init__(self, concepts=None):
        '''
        :param concepts: a binary np.array encoding the concepts practiced by this exercise.
        Could be one-hot for simple model, so each exercise practices exactly one concept.
        '''
        # if concepts is None, a random concept is chosen.
        if concepts:
            self.concepts = concepts
        else:
            # create a one hot vector for concepts
            self.concepts = np.zeros((N_CONCEPTS,))
            self.concepts[random.randint(0, N_CONCEPTS - 1)] = 1


class Student(object):
    def __init__(self, initial_knowledge=None):
        if initial_knowledge:
            self.knowledge = initial_knowledge
        else:
            self.knowledge = np.zeros((N_CONCEPTS,))

        # other potential member variables
        # self.motivation = 1

    def practice(self, exercise):
        '''

        :param exercise: an Exercise object
        :return:
        '''
        pass


def main():
    tree = ConceptDependencyTree()
    tree.init_default_tree(n=11)
    tree.print_edges()


if __name__ == "__main__":
    main()


