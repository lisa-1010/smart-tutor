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
from collections import defaultdict, deque


# Params / Constants
N_CONCEPTS = 10
N_EXERCISES = 100
P_TRANS_SATISFIED = 0.5
P_TRANS_NOT_SATISFIED = 0.0


class ConceptDependencyTree(object):
    def __init__(self):
        self.root = None
        self.children = defaultdict(list) # edges go from parent (e.g. prerequisite) to child
        self.parents = defaultdict(list)
        self.prereq_map = defaultdict(set)


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
                self.children[i].append(2 * i + 1)
                self.parents[2 * i + 1].append(i)
            else:
                # for leaf nodes, add a pseudo edge pointing to -1.
                self.children[i].append(-1)
            if 2 * i + 2 < n:
                self.children[i].append(2 * i + 2)
                self.parents[2 * i + 2].append(i)
        self._create_prereq_map()


    def _create_prereq_map(self):
        queue = deque()
        queue.append(self.root)
        while(True):
            cur = queue.popleft()
            self._add_prereqs(cur)
            children = self.edges[cur]
            queue.extend(children)


    def _add_prereqs(self, cur):
        # get parents of cur
        parents = self.parents[cur]
        self.prereq_map[cur].add(parents)

        for p in parents:
            self.prereq_map[cur].add(self.prereq_map[p])


    def print_edges(self):
        print self.edges

    def get_prereqs(self, concept):
        prereqs = np.zeros((N_CONCEPTS,))
        for p in self.prereq_map[concept]:
            prereqs[p] = 1
        return prereqs



concept_dep_tree = ConceptDependencyTree()
concept_dep_tree.init_default_tree(n=N_CONCEPTS)


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


    def do_exercise(self, ex):
        '''
        :param ex: an Exercise object
        :return:
        '''
        if self._fulfilled_prereqs(ex):
            return 1 if random.random() <= P_TRANS_SATISFIED else 0
        else:
            return 1 if random.random() <= P_TRANS_NOT_SATISFIED else 0


    def _fulfilled_prereqs(self, ex):
        '''
        for each concept tested in the exercise, check if all prereqs are fulfilled.
        if prereqs for at least one concept are not fulfilled, then function returns False.
        :return: bool
        '''
        for c in ex.concepts:
            #
            if c == 1:
                prereqs = concept_dep_tree.get_prereqs(c)
                if np.sum(np.multiply(self.knowledge, prereqs)) == 0:
                    return False
        return True

def main():
    tree = ConceptDependencyTree()
    tree.init_default_tree(n=11)
    tree.print_edges()
    s = Student()

if __name__ == "__main__":
    main()


