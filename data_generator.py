# data_generator.py
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
# USAGE: from data_generator import *

import numpy as np
import random
import pickle
from collections import defaultdict, deque, Counter


# Params / Constants
N_CONCEPTS = 10
# N_EXERCISES = 100

# P_TRANS_SATISFIED = 0.5
# P_TRANS_NOT_SATISFIED = 0.1


class ConceptDependencyGraph(object):
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

concept_dep_tree = ConceptDependencyGraph()
concept_dep_tree.init_default_tree(n=N_CONCEPTS)

class Exercise(object):
    def __init__(self, concepts=None):
        '''
        :param concepts: a binary np.array encoding the concepts practiced by this exercise.
        Could be one-hot for simple model, so each exercise practices exactly one concept.
        '''
        # if concepts is None, a random concept is chosen.
        if concepts != None:
            self.concepts = concepts
        else:
            # create a one hot vector for concepts
            self.concepts = np.zeros((N_CONCEPTS,))
            self.concepts[random.randint(0, N_CONCEPTS - 1)] = 1


class Student(object):

    def __init__(self, p_trans_satisfied=0.5, p_trans_not_satisfied=0.0, initial_knowledge=None):
        self.p_trans_satisfied = p_trans_satisfied
        self.p_trans_not_satisfied = p_trans_not_satisfied
        if initial_knowledge != None:
            self.knowledge = initial_knowledge
        else:
            self.knowledge = np.zeros((N_CONCEPTS,))

        # other potential member variables
        # self.motivation = 1


    def do_exercise(self, ex):
        '''
        Simulates solving the provided exercise.
        :param ex: an Exercise object.
        :return: Returns 1 if student solved it correctly, 0 otherwise.
        '''
        if self._fulfilled_prereqs(ex):
            # print("P trans satisfied_{}".format(self.p_trans_satisfied))
            if np.random.random() <= self.p_trans_satisfied:
                for c in xrange(len(ex.concepts)):
                    if ex.concepts[c] == 1:
                        self.knowledge[c] = 1
                return 1
            else:
                return 0
        else:
            return 1 if random.random() <= self.p_trans_not_satisfied else 0


    def _fulfilled_prereqs(self, ex):
        '''
        for each concept tested in the exercise, check if all prereqs are fulfilled.
        if prereqs for at least one concept are not fulfilled, then function returns False.
        :return: bool
        '''
        for i in xrange(len(ex.concepts)):
            c = ex.concepts[i]
            if c == 1:
                prereqs = concept_dep_tree.get_prereqs(i)
                if np.sum(np.multiply(self.knowledge, prereqs)) != np.sum(prereqs):
                    return False
        return True


def generate_student_sample(seqlen=100, exercise_seq=None, initial_knowledge=None, verbose=True):
    '''

    :param seqlen: number of exercises the student will do.
    :param exercise_seq: Sequence of exercises. list of exercise objects.
        If None, this function will generate the default sequence [0 .. seqlen - 1]
    :param initial_knowledge: initial knowledge of student. If None, will be set to 0 for all concepts.
    :return:
    '''

    initial_knowledge = np.zeros((N_CONCEPTS,))
    initial_knowledge[0] = 1
    s = Student(initial_knowledge=initial_knowledge)

    if not exercise_seq:
        exercise_seq = []
        for i in xrange(seqlen):
            concepts = np.zeros((N_CONCEPTS,))
            concepts[i % N_CONCEPTS] = 1
            ex = Exercise(concepts=concepts)
            exercise_seq.append(ex)

    # Go through sequence of exercises and record whether student solved each or not
    student_performance = []
    n_exercises_to_mastery = -1
    for i, ex in enumerate(exercise_seq):
        result = s.do_exercise(ex)
        student_performance.append(result)
        # print s.knowledge
        if np.sum(s.knowledge) == N_CONCEPTS:
            # if verbose and n_exercises_to_mastery == -1:
            n_exercises_to_mastery = i + 1
            break
    # print exercise_seq
    print student_performance
    if n_exercises_to_mastery != -1:
        print "learned all concepts after {} exercises.".format(n_exercises_to_mastery)
        student_sample = zip(exercise_seq[:n_exercises_to_mastery], student_performance)
    else:
        print ("Did not learn all concepts after doing {} exercises.".format(len(exercise_seq)))
        student_sample = zip(exercise_seq, student_performance)

    return student_sample



def generate_data(n_students=100, seqlen=100, filename=None):
    # generate sequences for n_students
    # each sequence of length
    # if save_to
    data = []

    print ("Generating data for {} students, with max sequence length {}.".format(n_students, seqlen))
    for i in xrange(n_students):
        student_sample = generate_student_sample(seqlen=seqlen, exercise_seq=None, initial_knowledge=None, verbose=True)
        data.append(student_sample)
    if filename:
        pickle.dump(data, open(filename, 'wb+'))
    # print data
    return data

def load_data(filename="synthetic_data/toy_small.pickle"):
    data = pickle.load(open(filename, 'rb+'))
    print len(data)
    # print data[0]


def main():
    # tree = ConceptDependencyTree()
    # tree.init_default_tree(n=11)
    # print tree.children
    # print tree.parents
    # print tree.prereq_map

    # generate_student_sample()

    # generate_data(n_students=1000, seqlen=50, filename="synthetic_data/1000_stud_max_50_ex.pickle")
    load_data("synthetic_data/1000_stud_max_50_ex.pickle")

if __name__ == "__main__":
    main()


