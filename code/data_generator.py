# data_generator.py
# @author: Lisa Wang
# @created: Jan 30 2017
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
# CURRENT STATUS: Working
#===============================================================================
# USAGE: from data_generator import *
# generate_data(n_students=5, seqlen=50, policy='expert', filename="synthetic_data/toy_expert.pickle")


from __future__ import absolute_import, division, print_function

# Python libraries
import numpy as np
import random
import pickle
import time
import copy
from collections import defaultdict, deque, Counter


# Custom Modules
from filepaths import *
from constants import *
import dataset_utils

# Import custom classes
from concept_dependency_graph import *
from student import *
from exercise import *


def fulfilled_prereqs(concept_tree, knowledge, concepts):
    '''
    for each concept tested in the exercise, check if all prereqs are fulfilled.
    if prereqs for at least one concept are not fulfilled, then function returns False.
    :return: bool
    '''

    for i in xrange(len(concepts)):
        c = concepts[i]
        if c == 1:
            prereqs = concept_tree.get_prereqs(i)
            if np.sum(np.multiply(knowledge, prereqs)) != np.sum(prereqs):
                return False
    return True


def choose_next_concept_with_expert_policy(concept_tree, knowledge, verbose=False):
    """
    Choose an exercise that the student has all the prerequisites for
    :param knowledge:
    :return: concepts:
    """
    concepts = np.zeros((N_CONCEPTS,))
    for i in xrange(N_CONCEPTS):
        if not knowledge[i]:
            # if student hasn't learned concept yet:
            concepts[i] = 1
            if fulfilled_prereqs(concept_tree, knowledge, concepts):
                # if verbose:
                #     print "chose exercise with concept {}".format(i)
                return concepts
            else:
                concepts[i] = 0
    concepts[0] = 1
    return concepts


def generate_student_sample(concept_tree, seqlen=100, exercise_seq=None, initial_knowledge=None, policy=None, verbose=False):
    '''
    :param concept_tree: Concept dependency graph
    :param seqlen: number of exercises the student will do.
    :param exercise_seq: Sequence of exercises. list of exercise objects.
        If None, this function will generate the default sequence [0 .. seqlen - 1]
        if exercise_seq provided, policy arg will be disregarded.
    :param initial_knowledge: initial knowledge of student. If None, will be set to 0 for all concepts.
    :param policy: if no exercise_seq provided, use the specified policy to generate exercise sequence.
    :param verbose: if True, print out debugging / progress statements
    :return: array of tuples, where each tuple consists of
    (exercise, 0 or 1 indicating success of student on that exercise, knowledge of student after doing exercise)
    Note that this array will have length seqlen, inclusive
    '''

    initial_knowledge = np.zeros((N_CONCEPTS,))
    initial_knowledge[0] = 1
    s = Student(initial_knowledge=initial_knowledge)

    # if not exercise_seq and policy == 'expert':
    #     return _generate_student_sample_with_expert_policy(student=s, seqlen=seqlen, verbose=verbose)

    if not exercise_seq and policy != 'expert':
        # for expert policy, we have to choose the next exercise online.
        exercise_seq = []
        for i in xrange(seqlen):
            concepts = np.zeros((N_CONCEPTS,))
            if policy == 'modulo':
                # choose exercise with modulo op. This imposes an ordering on exercises.
                concepts[i % N_CONCEPTS] = 1
            elif policy == 'random':
                # choose one random concept for this exercise
                concepts[np.random.randint(N_CONCEPTS)] = 1
            ex = Exercise(concepts=concepts)
            exercise_seq.append(ex)

    # Go through sequence of exercises and record whether student solved each or not
    student_performance = []
    student_knowledge = []
    n_exercises_to_mastery = -1
    exercises = [] # so we can store sequence of exercises as numpy arrays (instead of arrays of exercise objects)
    for i in xrange(seqlen):
        # print (s.knowledge)
        if policy == 'expert':
            concepts = choose_next_concept_with_expert_policy(concept_tree, s.knowledge, verbose=verbose)
            ex = Exercise(concepts=concepts)
        else:
            ex = exercise_seq[i]
        result = s.do_exercise(concept_tree, ex)
        exercises.append(ex.concepts) # makes the assumption that an exercise is equivalent to the concepts it practices)
        student_performance.append(result)
        student_knowledge.append(copy.deepcopy(s.knowledge))
        if np.sum(s.knowledge) == N_CONCEPTS and n_exercises_to_mastery == -1:
            # if verbose and n_exercises_to_mastery == -1:
            n_exercises_to_mastery = i + 1
    if verbose:
        if n_exercises_to_mastery != -1:
            print ("learned all concepts after {} exercises.".format(n_exercises_to_mastery))
        else:
            print ("Did not learn all concepts after doing {} exercises.".format(seqlen))
    # print (student_knowledge)
    student_sample = zip(exercises, student_performance, student_knowledge)
    return student_sample


def generate_data(concept_tree, n_students=100, seqlen=100, policy='modulo', filename=None, verbose=False):
    """
    :param concept_tree: Concept dependency graph
    :param n_students: number of students / samples to generate data for
    :param seqlen: max length of exercises for a student. if student learns all concepts, sequence can be shorter.
    :param policy: which policy to use to generate data. can be 'expert', 'modulo', 'random'
    :param filename: where to store the generated data. If None, will not save to file.
    :param verbose: if True, prints debugging statements
    :return:
    """

    data = []
    print ("Generating data for {} students, with max sequence length {}.".format(n_students, seqlen))
    for i in xrange(n_students):
        if verbose:
            print ("Creating sample for {}th student".format(i))
        student_sample = generate_student_sample(concept_tree, seqlen=seqlen, exercise_seq=None, initial_knowledge=None,
                                                 policy=policy, verbose=verbose)
        data.append(student_sample)
    if filename:
        pickle.dump(data, open(filename, 'wb+'))
    # print data
    return data


# def load_data(filename=None):
#     data = pickle.load(open(filename, 'rb+'))
#     return data


def get_data_stats(data):
    average_n_exercises = 0
    for i, sample in enumerate(data):
        n_exercises = len(sample)
        average_n_exercises += n_exercises
    average_n_exercises /= float(len(data))
    print ("Average number of exercises needed to get all concepts learned: {}".format(average_n_exercises))


def make_toy_data(concept_tree):
    filename = "toy.pickle"
    generate_data(concept_tree, n_students=5, seqlen=50, filename= "{}{}".format(SYN_DATA_DIR, filename))


def load_toy_data():
    filename = "toy.pickle"
    data = dataset_utils.load_data(filename= "{}{}".format(SYN_DATA_DIR, filename))
    print ("Loaded data. # samples:  {}".format(len(data)))


def main_test():
    """
    Run this to test this module.
    - Tests ConceptDependencytree
    - Generates sample for a single student using three different policies
    - Generates toy data set with 5 students
    - Loads generated toy data set
    """
    concept_tree = ConceptDependencyGraph()
    concept_tree.init_default_tree(n=11)
    print (concept_tree.children)
    print (concept_tree.parents)
    print (concept_tree.prereq_map)
    print ("Generate one sample using expert policy. ")
    generate_student_sample(concept_tree, policy='expert', verbose=True)
    print ("Generate one sample using random policy. ")
    generate_student_sample(concept_tree, policy='random', verbose=True)
    print ("Generate one sample using modulo policy. ")
    generate_student_sample(concept_tree, policy='modulo', verbose=True)

    make_toy_data(concept_tree)
    load_toy_data()


def init_synthetic_data():
    """
    Run this to generate the default synthetic data sets.
    :return:
    """
    concept_tree = ConceptDependencyGraph()
    concept_tree.init_default_tree(n=N_CONCEPTS)
    print ("Initializing synthetic data sets...")
    n_students = 1000
    seqlen = 100
    for policy in ['random', 'expert', 'modulo']:
        filename = "{}stud_{}seq_{}.pickle".format(n_students, seqlen, policy)
        generate_data(concept_tree, n_students=n_students, seqlen=seqlen, policy=policy, filename="{}{}".format(SYN_DATA_DIR, filename))
    print ("Data generation completed. ")

if __name__ == "__main__":
    main_test()
    # init_synthetic_data()


