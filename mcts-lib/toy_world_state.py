from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import rv_discrete, entropy
from copy import deepcopy


class ToyWorldAction(object):
    '''
    Up, down, left, right
    '''
    def __init__(self, action):
        self.action = action
        self._hash = 10*(action[0]+2) + action[1]+2

    def __hash__(self):
        return int(self._hash)

    def __eq__(self, other):
        return (self.action == other.action).all()

    def __str__(self):
        return str(self.action)

    def __repr__(self):
        return str(self.action)


class ToyWorld(object):
    def __init__(self, size, goal, manual):
        self.size = np.asarray(size)
        self.goal = np.asarray(goal)
        self.manual = manual


class ToyWorldState(object):
    '''
    Belief is just the current position/state
    '''
    def __init__(self, pos, world, belief=None):
        self.pos = pos
        self.world = world
        self.actions = [ToyWorldAction(np.array([0, 1])),
                        ToyWorldAction(np.array([0, -1])),
                        ToyWorldAction(np.array([1, 0])),
                        ToyWorldAction(np.array([-1, 0]))]
        if belief:
            self.belief = belief
        else:
            self.belief = pos

    def _correct_position(self, pos):
        upper = np.min(np.vstack((pos, self.world.size)), 0)
        return np.max(np.vstack((upper, np.array([0, 0]))), 0)

    def perform(self, action):

        # build next state
        pos = self._correct_position(self.pos + action.action)

        return ToyWorldState(pos, self.world)

    def real_world_perform(self, action):
        pos = self._correct_position(self.pos + action.action)

        return ToyWorldState(pos, self.world)

    def is_terminal(self):
        return False

    def __eq__(self, other):
        return (self.pos == other.pos).all()

    def __hash__(self):
        return int(self.pos[0]*100 + self.pos[1])

    def __str__(self):
        return str(self.pos)

    def __repr__(self):
        return str(self.pos)

    def reward(self, parent, action):
        if (self.pos == self.world.goal).all():
            print("g", end="")
            return 100
        else:
            reward = -1
            return reward