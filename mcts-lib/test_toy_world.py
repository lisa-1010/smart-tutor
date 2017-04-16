# Edited from original to test out various things
from __future__ import division

from toy_world_state import *

import random

from graph import (depth_first_search, _get_actions_and_states, StateNode)
from mcts import *
from utils import rand_max

import tree_policies
import default_policies
import backups

def toy_world_root():
    world = ToyWorld((3, 3), (3, 3), np.array([3, 3]))
    state = ToyWorldState((0, 0), world)
    root = StateNode(None, state)
    return root, state

def test_n_run_uct_search(toy_world_root, gamma, nrollouts, nsteps):
    root, state = toy_world_root
    random.seed()
    
    #rollout_policy = default_policies.immediate_reward
    rollout_policy = default_policies.RandomKStepRollOut(3)
    uct = MCTS(tree_policies.UCB1(1.41), rollout_policy,
               backups.Bellman(gamma))
    
    
    for i in range(nsteps):
        print('Step {}'.format(i))
        best_action = uct(root, n=nrollouts)
        print('Current state: {}'.format(str(root.state)))
        print(best_action)
        
        # act in the real environment
        new_root = root.children[best_action].sample_state(real_world=True)
        new_root.parent = None # cutoff the rest of the tree
        root = new_root
        print('Next state: {}'.format(str(new_root.state)))


if __name__ == '__main__':
    pass
    #test_perform()
    test_n_run_uct_search(toy_world_root(), 0.95, 100, 10)