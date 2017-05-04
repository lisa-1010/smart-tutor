import random


def immediate_reward(state_node):
    """
    Estimate the reward with the immediate return of that state.
    :param state_node:
    :return:
    """
    #DANIEL: edited the reward function to get rid of the parent argument and will just be a function of the current state
    return state_node.state.reward()


class RandomKStepRollOut(object):
    """
    Estimate the reward with the sum of returns of a k step rollout
    """
    def __init__(self, k):
        self.k = k

    def __call__(self, state_node):
        self.current_k = 0

        def stop_k_step(state):
            self.current_k += 1
            return self.current_k >= self.k or state.is_terminal()

        return _roll_out(state_node, stop_k_step)


def random_terminal_roll_out(state_node):
    """
    Estimate the reward with the sum of a rollout till a terminal state.
    Typical for terminal-only-reward situations such as games with no
    evaluation of the board as reward.

    :param state_node:
    :return:
    """
    def stop_terminal(state):
        return state.is_terminal()

    return _roll_out(state_node, stop_terminal)


def _roll_out(state_node, stopping_criterion):
    #DANIEL fixed so that it still returns the correct reward when starting on a terminal state
    #DANIEL: edited the reward function to get rid of the parent argument and will just be a function of the current state
    reward = 0
    state = state_node.state
    while True:
        reward += state.reward()
        if stopping_criterion(state):
            break
        else:
            action = random.choice(state.actions)
            state = state.perform(action)
    return reward
