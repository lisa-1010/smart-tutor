# student_model.py
# @author: Lisa Wang
# @created: Jan 30 2016
#
#===============================================================================
# DESCRIPTION:
# Ground truth student model for data generation.
# 1. n concepts (e.g. 10)
# 2. k exercises (e.g. 1000)
#
# ### Student Model
# At any time t, a student s can be represented by the concepts she knows.
# Hence, s is a n-dim vector, where each index i corresponds to concept i.

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
        while len(queue) > 0:
            cur = queue.popleft()
            self._add_prereqs(cur)
            children = self.children[cur]
            queue.extend(children)


    def _add_prereqs(self, cur):
        # get parents of cur
        parents = self.parents[cur]

        self.prereq_map[cur] = self.prereq_map[cur].union(set(parents))

        for p in parents:
            self.prereq_map[cur] = self.prereq_map[cur].union(self.prereq_map[p])


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



def generate_student_sample():
    pass

def generate_data(n_students=1000, seqlen=50):
    # generate sequences for n_students
    # each sequence of length
    pass

def main():
    tree = ConceptDependencyTree()
    tree.init_default_tree(n=11)
    print tree.children
    print tree.parents
    print tree.prereq_map
    # s = Student()

if __name__ == "__main__":
    main()


