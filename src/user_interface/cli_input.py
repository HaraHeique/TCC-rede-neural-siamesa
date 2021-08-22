# -*- coding: utf-8 -*-

"""
    It is responsible to handle user input information about training and prediction of the Neural Network,
    such as files to read, model variables and so on.
"""

import os
import src.user_interface.cli_output as uo
from src.enums.Stage import Stage
from src.enums.SimilarityMeasureType import SimilarityMeasureType
from src.enums.NeuralNetworkType import NeuralNetworkType
from src.enums.DatasetType import DatasetType
from src.enums.WordEmbeddingType import WordEmbeddingType

_DATA_FILES_PATH = os.path.dirname(os.path.abspath("src")) + "/data"
_DATA_FILES_TRAINING_PATH = _DATA_FILES_PATH + "/training"
_DATA_FILES_PREDICTION_PATH = _DATA_FILES_PATH + "/prediction"


def choose_stage():
    while True:
        option_chosen = input("What would you like to do?\n 0 - Leave\n 1 - Data Structure\n 2 - Train\n 3 - Predict\n 4 - Experiment\n")

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


def insert_word_embedding():
    while True:
        word_embedding_choosen = input("Choose the Word Embedding to represent words:\n 1 - Word2vec English Wikipedia\n 2 - Word2vec Google News\n 3 - Glove Wikipedia + Gigaword 5\n 4 - Glove Common Crawl uncased\n")

        if __try_parse_int_positive(word_embedding_choosen) and __is_word_embedding_type_valid(int(word_embedding_choosen)):
            return WordEmbeddingType(int(word_embedding_choosen))


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


def insert_dataset_type():
    input_message = "Choose the Dataset version:\n 1 - Raw (contain stopwords and no lemmatization)\n 2 - Without stopwords and no lemmatization\n 3 - Without stopwords and with lemmatization\n"

    while True:
        dataset_type = input(input_message)

        if __try_parse_int_positive(dataset_type) and __is_dataset_type_valid(int(dataset_type)):
            return DatasetType(int(dataset_type))


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


def insert_number_of_authors():
    while True:
        number_authors = input("Enter the number of authors to structure data: ")

        if __try_parse_int_positive(number_authors) and int(number_authors) > 0:
            return int(number_authors)


def insert_number_epochs():
    while True:
        number_epochs = input("Enter the number of epochs to train: ")

        if __try_parse_int_positive(number_epochs) and int(number_epochs) > 0:
            return int(number_epochs)


def insert_max_seq_length(isTraining=True):
    process_message = "train" if isTraining else "predict"

    while True:
        max_seq_length = input("Enter the number of max sequence length to {}: ".format(process_message))

        if __try_parse_int_positive(max_seq_length) and int(max_seq_length) > 0:
            return int(max_seq_length)


def insert_n_rounds_experiments():
    while True:
        number_experiments = input("Enter the number of rounds to run the experiments: ")

        if __try_parse_int_positive(number_experiments) and int(number_experiments) > 0:
            return int(number_experiments)


def insert_hyperparameters_variables():
    hyperparameters = {}

    hyperparameters['neural_network_type'] = insert_neural_network_type()
    uo.break_lines(1)
    hyperparameters['similarity_measure_type'] = insert_similarity_measure_type()
    uo.break_lines(1)
    hyperparameters['percent_validation'] = insert_percent_validation()
    uo.break_lines(1)
    hyperparameters['n_epochs'] = insert_number_epochs()
    uo.break_lines(1)
    hyperparameters['max_seq_length'] = insert_max_seq_length()
    uo.break_lines(1)

    return hyperparameters


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
    if 0 <= option <= 4:
        return True

    print("Please insert a option 0, 1, 2, 3 or 4")
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


def __is_dataset_type_valid(dataset_type):
    if 1 <= dataset_type <= 3:
        return True

    print("Dataset version is invalid. Try again.")
    return False


def __is_word_embedding_type_valid(word_embedding):
    if 1 <= word_embedding <= 4:
        return True

    print("Word embedding version is invalid. Try again.")
    return False
