# -*- coding: utf-8 -*-

"""
    It is responsible to handle user input information about training and prediction of the Neural Network,
    such as files to read, model variables and so on.
"""

import os
from src.enums.Stage import Stage
from src.enums.SimilarityMeasureType import SimilarityMeasureType
from src.enums.NeuralNetworkType import NeuralNetworkType

_DATA_FILES_PATH = os.path.dirname(os.path.abspath("src")) + "/data"
_DATA_FILES_TRAINING_PATH = _DATA_FILES_PATH + "/training"
_DATA_FILES_PREDICTION_PATH = _DATA_FILES_PATH + "/prediction"


def choose_stage():
    while True:
        option_chosen = input("What would you like to do?\n 0 - Leave\n 1 - Data Structure\n 2 - Train\n 3 - Predict\n")

        if __try_parse_int_positive(option_chosen) and __check_chosen_option(int(option_chosen)):
            return Stage(int(option_chosen))


def insert_training_filename():
    while True:
        filename = input("Enter the name of the training file located inside the /data/training: ")

        if __check_file_existence(_DATA_FILES_TRAINING_PATH, filename):
            return os.path.join(_DATA_FILES_TRAINING_PATH, filename)


def insert_prediction_filename():
    while True:
        filename = input("Enter the name of the prediction file located inside the /data/prediction: ")

        if __check_file_existence(_DATA_FILES_PREDICTION_PATH, filename):
            return os.path.join(_DATA_FILES_PREDICTION_PATH, filename)


def insert_neural_network_type():
    while True:
        network_type = input("Choose the Neural Network Type:\n 1 - LSTM\n 2 - CNN\n")

        if __try_parse_int_positive(network_type) and __is_network_type_valid(int(network_type)):
            return NeuralNetworkType(int(network_type))


def insert_similarity_measure_type():
    while True:
        similarity_type = input("Choose the Similarity Measure:\n 1 - MANHATTAN\n 2 - EUCLIDEAN\n 3 - COSINE\n")

        if __try_parse_int_positive(similarity_type) and __is_similarity_type_valid(int(similarity_type)):
            return SimilarityMeasureType(int(similarity_type))


def insert_percent_validation():
    while True:
        percent = input("Enter the percent of validation between 0% and 30%: ")

        if __try_parse_int_positive(percent) and __is_percent_valid(float(percent)):
            return float(percent)


def insert_number_sentences(input_message=None):
    default_message = "Enter the number of sentences of each author to structure data: "
    input_message = input_message if input_message is not None else default_message

    while True:
        number_sentences = input(input_message)

        if __try_parse_int_positive(number_sentences) and \
           __try_parse_even_number(number_sentences) and \
           int(number_sentences) > 0:
            return int(number_sentences)


def insert_number_epochs():
    while True:
        number_epochs = input("Enter the number of epochs to train: ")

        if __try_parse_int_positive(number_epochs) and int(number_epochs) > 0:
            return int(number_epochs)


def insert_max_seq_length():
    while True:
        max_seq_length = input("Enter the number of max sequence length to train: ")

        if __try_parse_int_positive(max_seq_length) and int(max_seq_length) > 0:
            return int(max_seq_length)


def __try_parse_int_positive(str_int):
    if not str_int.isdigit() or int(str_int) < 0:
        print("Please insert a valid and positive integer number")
        return False

    return True


def __try_parse_even_number(str_int):
    if not str_int.isdigit() or int(str_int) % 2 != 0:
        print("Please insert a valid and even positive integer number")
        return False

    return True


def __check_chosen_option(option):
    if 0 <= option <= 3:
        return True

    print("Please insert a option 0, 1, 2 or 3")
    return False


def __check_file_existence(directory, filename):
    if os.path.isfile(os.path.join(directory, filename)):
        return True

    print("The filename {0} does not exist. Try again".format(filename))
    return False


def __is_percent_valid(value):
    if 0 <= value <= 30:
        return True

    print("Percentage must be between 0% and 30%")
    return False


def __is_network_type_valid(network_type):
    if network_type == 1 or network_type == 2:
        return True

    print("Neural Network Type is invalid. Try again.")
    return False


def __is_similarity_type_valid(similarity_type):
    if 1 <= similarity_type <= 3:
        return True

    print("Similarity Measure is invalid. Try again.")
    return False
