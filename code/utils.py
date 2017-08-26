# helper functions


import os

def check_if_path_exists_or_create(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:  # Guard against race condition
            return False
    return True

############ converting histories to indices ##############################

def num_histories(index_base, horizon):
    '''
    Return the number of possible histories of the given horizon.
    '''
    return index_base ** horizon

def action_ob_encode(n_concepts, action, ob):
    '''
    Encode (action,ob) tuple as a unique number.
    '''
    return n_concepts * ob + action

def history_ix_append(n_concepts, history_ix, next_branch):
    '''
    History is encoded where the last tuple is the least significant digit.
    '''
    return history_ix * n_concepts * 2 + next_branch