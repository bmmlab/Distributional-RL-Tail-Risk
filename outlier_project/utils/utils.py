from pathlib import Path
import json
import numpy as np


def create_folder_if_not_exists(**kwargs) -> None:
    data_path = "../simulation_data/" + "seed_" + str(kwargs["seed"]) + "/" + kwargs["which_method"] + "/" + kwargs[
        "which_dist"]

    if not Path(data_path).exists():
        Path(data_path).mkdir(parents=True)
    else:
        pass


def save_simulation_data_to_json(seed: int, which_method: str, which_dist: str, which_simulation: int, result_dict: dict):
    create_folder_if_not_exists(which_dist=which_dist, which_method=which_method, seed=seed)
    with open("../simulation_data/" + "seed_" + str(seed) + "/" + which_method + "/" + which_dist + "/" +
              str(which_simulation) + ".json", 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)


def save_configuration_to_json(c, seed):
    data_path = "../simulation_data/" + "seed_" + str(seed) + "/"

    if not Path(data_path).exists():
        Path(data_path).mkdir(parents=True)
    else:
        pass

    with open(data_path + "configuration.json", 'w', encoding='utf-8') as f:
        json.dump(c.__dict__, f, ensure_ascii=False, indent=4)


def initialize_record_distribution(states, actions, empty_atoms):
    """
    initialize a distribution memory
    :param states: a list of states
    :param actions: a list of actions
    :param empty_atoms: a list of empty atoms
    :return: a dictionary to record values
    """
    dist = {}

    atom_shape = empty_atoms.shape

    # initial values, equal probability distribution
    val = np.full(shape=atom_shape, fill_value=1.0 / float(atom_shape[0]))

    for i in states:
        dist[i] = {}
        for j in actions:
            dist[i][j] = val.copy()

    return dist


def initialize_distribution(states, actions):
    """
    initialize a distribution memory
    :param states: a list of states
    :param actions: a list of actions
    :return: a dictionary to record values
    """
    dist = {}

    for i in states:
        dist[i] = {}
        for j in actions:
            dist[i][j] = [0.0]

    return dist


def dict_to_list(states, dictionary):
    """
    :param states: no. of states
    :param dictionary: input distribution
    :return: a list of values for each states
    """
    rows = []
    for i in states:
        rows.append(list(dictionary[i].values()))
    return rows.copy()
