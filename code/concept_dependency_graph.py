# concept_dependency_graph.py
# @author: Lisa Wang
# @created: Apr 25 2017
#
#===============================================================================
# DESCRIPTION:
# Defines a concept dependency graph data structure as a Python object.
#===============================================================================
# CURRENT STATUS: Working
#===============================================================================
# USAGE: from concept_dependency_graph import *


from __future__ import absolute_import, division, print_function

# Python libraries
import numpy as np
from collections import defaultdict, deque, Counter

# Custom Modules
from constants import *


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
