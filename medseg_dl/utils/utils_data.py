from medio import parse_dir_tf
import os
import json
import logging
import yaml


def load_sets(dir_files_in, dir_files_out, path_parser_cfg, set_split, b_recreate=False):
    """ load training and evaluation set """

    # fetch training and eval filenames
    file_train = os.path.join(dir_files_out, 'filenames_train.json')
    file_eval = os.path.join(dir_files_out, 'filenames_eval.json')

    # generate if not available or to be recreated
    if b_recreate or not os.path.isfile(file_train) or not os.path.isfile(file_eval):
        if not os.path.isdir(dir_files_in):
            raise ValueError('Can not generate dataset for non-existing directory')
        gen_sets(dir_files_out, dir_files_in, path_parser_cfg=path_parser_cfg, set_split=set_split)

    # load data
    with open(file_train, 'r') as file:
        filenames_train = json.load(file)

    with open(file_eval, 'r') as file:
        filenames_eval = json.load(file)

    return filenames_train, filenames_eval


def gen_sets(dir_out, dir_data, path_parser_cfg, set_split=120):
    """ generate training and evaluation set and save them """

    file_train = os.path.join(dir_out, 'filenames_train.json')
    file_eval = os.path.join(dir_out, 'filenames_eval.json')

    # generate data dict containing file lists
    data, patterns = fetch_sets(dir_data, path_parser_cfg=path_parser_cfg)

    # perform dataset split on subject level
    subjects = list(data.keys())
    subjects_train = subjects[:set_split]
    subjects_eval = subjects[set_split:]

    # generate actual lists used during training / testing
    filenames_train = fetch_paths(data, patterns, subjects_train)
    filenames_eval = fetch_paths(data, patterns, subjects_eval)

    with open(file_train, 'w') as file:
        json.dump(filenames_train, file)

    with open(file_eval, 'w') as file:
        json.dump(filenames_eval, file)


def fetch_paths(data_dict, patterns, chosen_subjects):
    """
    extract lists from fetched dictionary
    :param data_dict:
    :param patterns: provides structure for converting dict to lists (could be self-derived, but would result in bloated code)
    :param chosen_subjects:
    :return:
    """

    code_categories = ['image', 'label']  # required to ensure deterministic category ordering
    # ensure category conventions of current code
    for category in code_categories:
        if category not in patterns.keys():
            raise ValueError(f'Current code expects category \'{category}\' within the passed patterns dict')

    paths = [[] for _ in range(len(code_categories))]
    for subject in chosen_subjects[:]:
        for idx_cat, category in enumerate(code_categories):
            paths[idx_cat].append([str(x) for x in data_dict[subject][category]])

    # zip into long lists for each sub-category
    for idx in range(len(paths)):
        paths[idx] = list(zip(*paths[idx]))

    return paths


def fetch_sets(dir_data, path_parser_cfg, b_verbose=True):
    """ fetch wanted sets """

    with open(path_parser_cfg, 'r') as file:
        patterns = yaml.load(file)

    logging.info(f'Fetching data with pattern {patterns}')
    data = parse_dir_tf.fetch_data(dir_data, patterns=patterns, b_verbose=b_verbose)

    return data, patterns
