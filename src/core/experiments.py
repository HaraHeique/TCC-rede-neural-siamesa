# -*- coding: utf-8 -*-

"""
    Contains a set of functions to run experiments of training and prediction process.
"""

import os
import pathlib
import statistics
import itertools
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import src.core.helper as helper
import src.core.training as training
import src.core.prediction as prediction

from datetime import datetime
from src.enums.SimilarityMeasureType import SimilarityMeasureType
from src.enums.NeuralNetworkType import NeuralNetworkType
from src.enums.DatasetType import DatasetType
from src.enums.WordEmbeddingType import WordEmbeddingType

__DATE_EXPERIMENT = datetime.now()


def run_experiments(n_rounds):
    configs_per_datasets = __get_defined_configs_per_dataset_type()

    for dataset_type in list(DatasetType):
        for config_dataset in configs_per_datasets[dataset_type]:
            n_epochs = config_dataset[0]
            max_seq_length = config_dataset[1]

            for round_number in range(1, n_rounds + 1, 1):
                hyperparameters = __get_defined_lstm_hyperparameters(n_epochs, max_seq_length, 1)
                model = __run_training_process_round(round_number, hyperparameters, dataset_type)
                __run_prediction_process_round(round_number, model, hyperparameters, dataset_type)

            __save_training_plot_performance_graph(n_rounds, dataset_type, max_seq_length, n_epochs)
            __save_prediction_authors_matrix_combinations(n_rounds, dataset_type, n_epochs, max_seq_length)


def __get_defined_lstm_hyperparameters(n_epochs, max_seq_length, gpus):
    hyperparameters_lstm = {
        'max_seq_length': max_seq_length,
        'percent_validation': 30,
        'n_epochs': n_epochs,
        'similarity_measure_type': SimilarityMeasureType.MANHATTAN,
        'neural_network_type': NeuralNetworkType.LSTM,
        'gpus': gpus,
        'embedding_dim': 300,
        'batch_size': 64 * gpus,
        'n_hidden': 128,
        'kernel_initializer': tf.keras.initializers.glorot_normal(seed=1),
        'kernel_regularizer': None,
        'bias_regularizer': None,
        'activity_regularizer': None,
        'activation': "softsign",
        'recurrent_activation': "hard_sigmoid",
        'dropout': 0.24697647633830466,
        'dropout_lstm': 0.8228431108922754,
        'recurrent_dropout': 0.06536980304050743,
        'activation_layer': "selu",
        'activation_dense_layer': "sigmoid",
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adadelta(learning_rate=0.1, rho=0.95, epsilon=1e-07, name='Adadelta',
                                                  clipnorm=1.5)
    }

    return hyperparameters_lstm


def __get_defined_configs_per_dataset_type():
    # based sentences distribution (mean, median and mean of mean and median)
    # The tuple is (n_epochs, max_seq_length)
    MOST_EPOCHS = 2

    return {
        DatasetType.RAW: [(MOST_EPOCHS, 17), (MOST_EPOCHS, 10), (MOST_EPOCHS, 14)],
        DatasetType.WITHOUT_SW: [(MOST_EPOCHS, 9), (MOST_EPOCHS, 5), (MOST_EPOCHS, 7)],
        DatasetType.WITHOUT_SW_WITH_LEMMA: [(MOST_EPOCHS, 9), (MOST_EPOCHS, 5), (MOST_EPOCHS, 7)]
    }


def __run_training_process_round(round_number, hyperparameters, dataset_type):
    word_embedding_type = WordEmbeddingType.WORD2VEC_GOOGLE_NEWS

    # Loading index vector and embedding matrix
    index_vector_filename = helper.get_index_vector_filename(dataset_type, word_embedding_type,
                                                             training_process=True)
    embedding_matrix_filename = helper.get_embedding_matrix_filename(dataset_type, word_embedding_type)

    training_dataframe = helper.load_index_vector_dataframe(index_vector_filename)
    embedding_matrix = helper.load_embedding_matrix(embedding_matrix_filename)

    # Data preparation and normalization
    validation_size = training.get_validation_size(training_dataframe, hyperparameters['percent_validation'])
    splited_data_training = training.split_data_train(training_dataframe)
    normalized_dataframe = training.define_train_and_validation_dataframe(splited_data_training['phrases'],
                                                                          splited_data_training['labels'],
                                                                          validation_size,
                                                                          hyperparameters['max_seq_length'])

    # Creating the model based on a similarity function/measure
    shared_model = training.define_shared_model(embedding_matrix, hyperparameters)

    model = training.define_model(shared_model, hyperparameters)
    training.compile_model(model, hyperparameters)
    training.show_summary_model(model)

    # Training the neural network based on model
    training_history = training.train_neural_network(model, normalized_dataframe, hyperparameters)

    # Save data in csv
    __save_training_experiments_results(round_number, training_history, hyperparameters['max_seq_length'], dataset_type)

    return model


def __save_training_experiments_results(round_number, training_history, max_seq_length, dataset_type):
    columns = ['date', 'round', 'max_seq_length', 'accuracy', 'val_accuracy', 'loss', 'val_loss']

    row_data = [
        __DATE_EXPERIMENT.strftime("%d/%m/%Y %H:%M:%S"), round_number, max_seq_length,
        training_history.history['accuracy'], training_history.history['val_accuracy'],
        training_history.history['loss'], training_history.history['val_loss']
    ]

    pathlib.Path(helper.get_experiments_path_directory_by_dataset(dataset_type)).mkdir(parents=True, exist_ok=True)
    path_file = os.path.join(helper.get_experiments_path_directory_by_dataset(dataset_type),
                             "rounds-model-training-results.csv")
    mode_file = 'a' if os.path.exists(path_file) else 'w'
    has_header = mode_file == 'w'

    dataframe = pd.DataFrame([row_data], columns=columns)
    dataframe.to_csv(path_file, index=False, mode=mode_file, header=has_header)


def __run_prediction_process_round(round_number, model, hyperparameters, dataset_type):
    word_embedding_type = WordEmbeddingType.WORD2VEC_GOOGLE_NEWS

    # Loading index vector and embedding matrix
    index_vector_filename = helper.get_index_vector_filename(dataset_type, word_embedding_type, training_process=False)
    df_prediction = helper.load_index_vector_dataframe(index_vector_filename)

    # Data preparation
    test_normalized_data = prediction.define_prediction_dataframe(df_prediction, hyperparameters['max_seq_length'])

    # Loading the model trained
    prediction.show_summary_model(model)

    # Predict data from the model trained
    prediction_result = prediction.predict_neural_network(model, test_normalized_data)
    authors_combinations_predictions = prediction.extract_authors_combinations_predictions(df_prediction,
                                                                                           prediction_result)

    # Save data in csv
    __save_prediction_experiments_results(round_number, authors_combinations_predictions, dataset_type,
                                          hyperparameters['max_seq_length'])


def __save_prediction_experiments_results(round_number, authors_combinations_predictions, dataset_type, max_seq_length):
    mean_authors_combinations = {}
    all_authors = []

    for author_combination, dic_pred in authors_combinations_predictions.items():
        pred_lst = dic_pred['y_pred']
        mean_authors_combinations[author_combination] = statistics.mean(pred_lst)
        all_authors += [dic_pred['author1'], dic_pred['author2']]

    all_authors = sorted(list(set(all_authors)))
    authors_prod_combinations = list(itertools.product(all_authors, all_authors))  # Combination in correct order

    columns = ['date', 'round', 'max_seq_length', 'author1', 'author2', 'mean_prediction']
    rows_data = []

    for author_combination in authors_prod_combinations:
        author1 = author_combination[0]
        author2 = author_combination[1]

        author_combination_key = "{}-{}".format(author1, author2)

        if author_combination_key not in mean_authors_combinations:
            author_combination_key = "{}-{}".format(author2, author1)

        mean_pred = mean_authors_combinations[author_combination_key]
        rows_data.append([__DATE_EXPERIMENT.strftime("%d/%m/%Y %H:%M:%S"), round_number, max_seq_length, author1, author2, mean_pred])

    pathlib.Path(helper.get_experiments_path_directory_by_dataset(dataset_type)).mkdir(parents=True, exist_ok=True)
    path_file = os.path.join(helper.get_experiments_path_directory_by_dataset(dataset_type),
                             "rounds-prediction-similarity-values.csv")
    mode_file = 'a' if os.path.exists(path_file) else 'w'
    has_header = mode_file == 'w'

    dataframe = pd.DataFrame(rows_data, columns=columns)
    dataframe.to_csv(path_file, index=False, mode=mode_file, header=has_header)


def __save_training_plot_performance_graph(n_rounds, dataset_type, max_seq_length, n_epochs):
    filename = os.path.join(helper.get_experiments_path_directory_by_dataset(dataset_type),
                            "rounds-model-training-results.csv")
    df_experiment = pd.read_csv(filename)

    accuracy_lst = [0.0] * n_epochs
    val_accuracy_lst = [0.0] * n_epochs
    loss_lst = [0.0] * n_epochs
    val_loss_lst = [0.0] * n_epochs

    for index, row in df_experiment.iterrows():
        df_date = row['date']
        df_max_seq_length = row['max_seq_length']

        if df_date != __DATE_EXPERIMENT.strftime("%d/%m/%Y %H:%M:%S") or df_max_seq_length != max_seq_length:
            continue

        row_accuracy_lst = __extract_vector_performance(row['accuracy'])
        row_val_accuracy_lst = __extract_vector_performance(row['val_accuracy'])
        row_loss_lst = __extract_vector_performance(row['loss'])
        row_val_loss_lst = __extract_vector_performance(row['val_loss'])

        for i in range(n_epochs):
            accuracy_lst[i] += row_accuracy_lst[i]
            val_accuracy_lst[i] += row_val_accuracy_lst[i]
            loss_lst[i] += row_loss_lst[i]
            val_loss_lst[i] += row_val_loss_lst[i]

    accuracy_lst = list(map(lambda val: val / n_rounds, accuracy_lst))
    val_accuracy_lst = list(map(lambda val: val / n_rounds, val_accuracy_lst))
    loss_lst = list(map(lambda val: val / n_rounds, loss_lst))
    val_loss_lst = list(map(lambda val: val / n_rounds, val_loss_lst))

    __save_graph_performance(accuracy_lst, val_accuracy_lst, loss_lst, val_loss_lst, dataset_type, n_epochs, max_seq_length)
    __save_final_results_training(accuracy_lst, val_accuracy_lst, loss_lst, val_loss_lst, dataset_type, max_seq_length)


def __extract_vector_performance(str_data_experiment):
    return [float(value) for value in str_data_experiment.replace('[', '').replace(']', '').split(',')]


def __save_graph_performance(accuracy_lst, val_accuracy_lst, loss_lst, val_loss_lst, dataset_type, epochs, max_seq_length):
    # Plot accuracy
    plt.subplot(211)
    plt.plot(accuracy_lst)
    plt.plot(val_accuracy_lst)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(212)
    plt.plot(loss_lst)
    plt.plot(val_loss_lst)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')

    base_filename = "history-graph-{date}-{epochs}-{max_seq_length}.png".format(
        date=__DATE_EXPERIMENT.strftime("%d_%m_%Y-%H_%M_%S"),
        epochs=epochs,
        max_seq_length=max_seq_length
    )
    filename = os.path.join(helper.get_experiments_path_directory_by_dataset(dataset_type), base_filename)

    plt.tight_layout(h_pad=1.0)
    plt.savefig(filename)


def __save_prediction_authors_matrix_combinations(n_rounds, dataset_type, epochs, max_seq_length):
    filename = os.path.join(helper.get_experiments_path_directory_by_dataset(dataset_type),
                            "rounds-prediction-similarity-values.csv")
    df_experiment = pd.read_csv(filename)
    dic_author_combination = {}

    all_authors = []

    for index, row in df_experiment.iterrows():
        df_date = row['date']
        df_max_seq_length = row['max_seq_length']

        if df_date != __DATE_EXPERIMENT.strftime("%d/%m/%Y %H:%M:%S") or df_max_seq_length != max_seq_length:
            continue

        author1 = row['author1']
        author2 = row['author2']
        all_authors += [author1, author2]

        author_combination_key = "{}-{}".format(author1, author2)

        if author_combination_key not in dic_author_combination:
            dic_author_combination[author_combination_key] = []

        dic_author_combination[author_combination_key].append(row['mean_prediction'])

    all_authors = sorted(list(set(all_authors)))
    table_data = []

    for author_combination, mean_pred_lst in dic_author_combination.items():
        mean_pred = statistics.mean(mean_pred_lst)
        table_data.append("{:.2f}".format(mean_pred))

    # CONFIGURING TABLE
    table_title = "Média total índices de similaridade"

    table_filename = helper.get_experiments_path_directory_by_dataset(dataset_type) + "/prediction-similarity-values-{date}-{epochs}-{max_seq_length}.png"
    table_filename = table_filename.format(
        date=__DATE_EXPERIMENT.strftime("%d_%m_%Y-%H_%M_%S"),
        epochs=epochs,
        max_seq_length=max_seq_length
    )

    prediction.save_authors_table_image(table_filename, table_title, table_data, all_authors)
    __save_final_results_prediction(dic_author_combination, dataset_type, max_seq_length)


def __save_final_results_training(accuracy_lst, val_accuracy_lst, loss_lst, val_loss_lst, dataset_type, max_seq_length):
    columns = ['date', 'max_seq_length', 'accuracy', 'val_accuracy', 'loss', 'val_loss']

    row_data = [
        __DATE_EXPERIMENT.strftime("%d/%m/%Y %H:%M:%S"), max_seq_length,
        accuracy_lst, val_accuracy_lst, loss_lst, val_loss_lst
    ]

    path_file = os.path.join(helper.get_experiments_path_directory_by_dataset(dataset_type),
                             "final-training-results.csv")
    mode_file = 'a' if os.path.exists(path_file) else 'w'
    has_header = mode_file == 'w'

    dataframe = pd.DataFrame([row_data], columns=columns)
    dataframe.to_csv(path_file, index=False, mode=mode_file, header=has_header)


def __save_final_results_prediction(dic_author_combination, dataset_type, max_seq_length):
    columns = ['date', 'max_seq_length', 'author1', 'author2', 'mean_prediction', 'standard_deviation']
    rows_data = []

    for author_combination, mean_pred_lst in dic_author_combination.items():
        author_combination_splitted = author_combination.split('-')

        rows_data.append([
            __DATE_EXPERIMENT.strftime("%d/%m/%Y %H:%M:%S"), max_seq_length,
            author_combination_splitted[0], author_combination_splitted[1],
            statistics.mean(mean_pred_lst), statistics.stdev(mean_pred_lst)
        ])

    path_file = os.path.join(helper.get_experiments_path_directory_by_dataset(dataset_type),
                             "final-prediction-results.csv")
    mode_file = 'a' if os.path.exists(path_file) else 'w'
    has_header = mode_file == 'w'

    dataframe = pd.DataFrame(rows_data, columns=columns)
    dataframe.to_csv(path_file, index=False, mode=mode_file, header=has_header)
