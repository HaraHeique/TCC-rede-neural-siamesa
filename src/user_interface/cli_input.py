# -*- coding: utf-8 -*-

"""
    It is responsible to handle user input information about training and prediction of the Neural Network,
    such as files to read, model variables and so on.
"""

import os
from src.enums.Stage import Stage

_DATA_FILES_PATH = os.path.dirname(os.path.abspath("src")) + "/data"
_DATA_FILES_TRAINING_PATH = _DATA_FILES_PATH + "/training"
_DATA_FILES_PREDICTION_PATH = _DATA_FILES_PATH + "/prediction"


def choose_stage():
    while True:
        option_chosen = input("What would you like to do?\n 0 - Leave\n 1 - Train\n 2 - Predict\n")

        if _try_parse_int_positive(option_chosen) and _check_chosen_option(int(option_chosen)):
            return Stage(int(option_chosen))


def insert_training_filename():
    while True:
        filename = input("Enter the name of the training file located inside the /data/training: ")

        if _check_file_existence(_DATA_FILES_TRAINING_PATH, filename):
            return os.path.join(_DATA_FILES_TRAINING_PATH, filename)


def insert_prediction_filename():
    while True:
        filename = input("Enter the name of the prediction file located inside the /data/prediction: ")

        if _check_file_existence(_DATA_FILES_PREDICTION_PATH, filename):
            return os.path.join(_DATA_FILES_PREDICTION_PATH, filename)


def insert_percent_validation():
    while True:
        percent = input("Enter the percent of validation between 0% and 30%: ")

        if _try_parse_int_positive(percent) and _is_percent_valid(float(percent)):
            return float(percent)


def _try_parse_int_positive(str_int):
    if not str_int.isdigit() or int(str_int) < 0:
        print("Please insert a valid and positive integer number")
        return False

    return True


def _check_chosen_option(option):
    if 0 <= option <= 2:
        return True

    print("Please insert a option 0, 1 or 2")
    return False


def _check_file_existence(directory, filename):
    if os.path.isfile(os.path.join(directory, filename)):
        return True

    print("The filename {0} does not exist. Try again".format(filename))
    return False


def _is_percent_valid(value):
    if 0 <= value <= 30:
        return True

    print("Percentage must be between 0% and 30%")
    return False
