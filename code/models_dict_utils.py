import os
import json

MODELS_DICT_PATH = 'models_dict.json'

def init_models_dict():
    all_models_dict = {}
    all_models_dict['dummy'] = {}
    # If no dummy is added, the dictionary would be empty and the json file would be empty, causing an error on load.
    with open(MODELS_DICT_PATH, 'a+') as fp:
        json.dump(all_models_dict, fp, sort_keys=True, indent=4)


def save_model_dict(model_id, model_dict):
    if not os.path.isfile(MODELS_DICT_PATH):
        # If the models dict file does not exist, create a new one with a dummy model.
        init_models_dict()

    all_models_dict = load_all_models_dict()
    all_models_dict[model_id] = model_dict

    with open(MODELS_DICT_PATH, 'w') as f:
        json.dump(all_models_dict, f, sort_keys=True, indent=4)


def load_model_dict(model_id):
    """
    Loads the dictionary with the properties for the model specified by model_id.
    :param model_id:
    :return: model_dict corresponding to model_id
    """
    all_models_dict = load_all_models_dict()
    assert (model_id in all_models_dict), "model_id could not be found in all_models_dict. "
    return all_models_dict[model_id]


def load_all_models_dict():
    cur_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cur_path, MODELS_DICT_PATH), 'r') as f:
        all_models_dict = json.load(f)
    return all_models_dict


def create_new_model_dict(n_timesteps, n_inputdim, n_hidden, n_classes, architecture):
    model_dict = {
        'n_timesteps': n_timesteps,
        'n_inputdim': n_inputdim,
        'n_hidden': n_hidden,
        'n_classes': n_classes,
        'architecture': architecture,
    }
    return model_dict


def diff_to_existing_model_dict(model_id, n_timesteps, n_inputdim, n_hidden, n_classes, architecture):
    model_dict = load_model_dict(model_id)
    new_model_dict = create_new_model_dict(n_timesteps, n_inputdim, n_hidden, n_classes, architecture)
    is_different = False
    for key in model_dict:
        if model_dict[key] != new_model_dict[key]:
            print ("{}: \t old value: {} \t new value: {}".format(key, model_dict[key], new_model_dict[key]))
            is_different = True
    if not is_different:
        print ("No differences found. Yay! ")


def model_exists_in_dict(model_id):
    """
    :param model_id: string.
    :return: bool, indicating whether model_id key was found in all_models_dict
    """
    all_models_dict = load_all_models_dict()
    return (model_id in all_models_dict)


def check_model_exists_or_create_new(model_id, n_timesteps, n_inputdim, n_hidden, n_classes, architecture):
    if not model_exists_in_dict(model_id):
        new_model_dict = create_new_model_dict(n_timesteps, n_inputdim, n_hidden, n_classes, architecture)
        save_model_dict(model_id, new_model_dict)
        print ("New model created. Ready to be loaded. ")
    else:
        print ("A model with the same model_id '{}' already exists. ".format(model_id))
        diff_to_existing_model_dict(model_id, n_timesteps, n_inputdim, n_hidden, n_classes, architecture)


if __name__ == '__main__':
    init_models_dict()