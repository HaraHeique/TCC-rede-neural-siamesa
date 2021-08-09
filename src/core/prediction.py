# -*- coding: utf-8 -*-

"""
    Contains a set of functions to predict the Neural Network.
"""

import os
import statistics
import itertools
import sklearn.metrics
import scipy.stats
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import src.core.helper as helper


def load_prediction_dataframe(filename):
    # Load training set
    prediction_dataframe = pd.read_csv(filename)
    for q in ['phrase1', 'phrase2']:
        prediction_dataframe[q + '_n'] = prediction_dataframe[q]

    return prediction_dataframe


def find_max_seq_length(train_dataframe):
    return helper.find_max_seq_length(train_dataframe)


def define_prediction_dataframe(prediction_dataframe, max_seq_length):
    # Split to dicts and append zero padding.
    x_prediction = helper.split_and_zero_padding(prediction_dataframe, max_seq_length)

    return x_prediction


def check_prediction_dataframe(x_prediction):
    # Make sure everything is ok
    assert x_prediction['left'].shape == x_prediction['right'].shape


def load_model(filename):
    model = tf.keras.models.load_model(filename)
    return model


def show_summary_model(model):
    model.summary()


def predict_neural_network(model, x_prediction):
    prediction = model.predict([x_prediction['left'], x_prediction['right']])

    return prediction


def save_authors_prediction_matrix(df_prediction, predictions, configs):
    dataset_type = configs['dataset_type']

    # CONFIGURING TABLE
    table_title = "Índices de Similaridade \n({network_type} - {similarity_type} - {dataset_type})".format(
        network_type=configs['neural_network_type'].name,
        similarity_type=configs['similarity_type'].name,
        dataset_type=dataset_type.name
    )

    table_filename = helper.get_results_path_directory_by_dataset(
        dataset_type) + "/prediction-similarity-values-{date}-{network_type}-{similarity_type}-{word_embedding_type}.png"
    table_filename = table_filename.format(
        date=configs['date'].strftime("%d_%m_%Y-%H_%M_%S"),
        network_type=configs['neural_network_type'].name,
        similarity_type=configs['similarity_type'].name,
        word_embedding_type=configs['word_embedding_type'].name
    )

    # Extracting and structuring data
    authors_combinations_predictions = __extract_authors_combinations_predictions(df_prediction, predictions)
    mean_authors_combinations = {}
    all_authors = []

    for author_combination, dic_pred in authors_combinations_predictions.items():
        pred_lst = dic_pred['y_pred']
        mean_authors_combinations[author_combination] = statistics.mean(pred_lst)
        all_authors += [dic_pred['author1'], dic_pred['author2']]

    all_authors = list(set(all_authors))

    # Create table data
    table_columns = all_authors
    table_data = []

    for author in all_authors:
        for author_combination, mean_pred in mean_authors_combinations.items():
            if author in author_combination.split('-'):
                table_data.append(mean_pred)

    # Save as a table image file
    __save_authors_table_image(table_filename, table_title, table_data, table_columns)

    # CONFIGURING CSV FILE
    base_filename = "prediction-similarity-values.csv"
    path_file = os.path.join(helper.get_results_path_directory_by_dataset(dataset_type), base_filename)
    mode_file = 'a' if os.path.exists(path_file) else 'w'
    has_header = mode_file == 'w'

    columns = [
        'date', 'neural_network_type',
        'similarity_type', 'embedding_type',
        'max_seq_length', 'author1', 'author2', 'mean_prediction'
    ]

    rows_data = []
    for author_combination, mean_pred in mean_authors_combinations.items():
        authors_splited = author_combination.split('-')
        rows_data.append([
            configs['date'].strftime("%d/%m/%Y %H:%M:%S"), configs['neural_network_type'].name,
            configs['similarity_type'].name, configs['word_embedding_type'].name,
            configs['max_seq_length'], authors_splited[0], authors_splited[1], mean_pred
        ])

    dataframe = pd.DataFrame(rows_data, columns=columns)
    dataframe.to_csv(path_file, index=False, mode=mode_file, header=has_header)


def save_authors_prediction_matrix_by_all_combinations(model, df_prediction, configs):
    # LOGIC OF PREDICTIONS OF COMBINATIONS AMONG PHRASES
    authors_index_vector_phrases = __extract_authors_phrases_vectors(df_prediction)
    authors = list(authors_index_vector_phrases.keys())
    authors_prod_combinations = list(itertools.product(authors, authors))
    all_combinations_mean_pred = []

    for author_combination in authors_prod_combinations:
        author1 = author_combination[0]
        author2 = author_combination[1]

        if author1 == author2:
            dic_combination_inputs = __combination_between_same_authors(authors_index_vector_phrases[author1])
        else:
            dic_combination_inputs = _combination_between_different_authors(
                authors_index_vector_phrases[author1],
                authors_index_vector_phrases[author2]
            )

        df_data = [dic_combination_inputs['author1'], dic_combination_inputs['author2']]
        df_author_combination = pd.DataFrame(df_data, columns=['phrase1_n', 'phrase2_n'])
        input_normalized_data = define_prediction_dataframe(df_author_combination, configs['max_seq_length'])

        predictions_list = __to_predictions_list(predict_neural_network(model, input_normalized_data))
        mean_pred = statistics.mean(predictions_list)
        all_combinations_mean_pred.append(mean_pred)

    # CONFIGURING TABLE
    dataset_type = configs['dataset_type']

    table_title = "Índices de similaridade entre todas combinações de frases\n({network_type} - {similarity_type} - {dataset_type})".format(
        network_type=configs['neural_network_type'].name,
        similarity_type=configs['similarity_type'].name,
        dataset_type=dataset_type.name
    )

    table_filename = helper.get_results_path_directory_by_dataset(
        dataset_type) + "/prediction-similarity-values-all-combinations-{date}-{network_type}-{similarity_type}-{word_embedding_type}.png"
    table_filename = table_filename.format(
        date=configs['date'].strftime("%d_%m_%Y-%H_%M_%S"),
        network_type=configs['neural_network_type'].name,
        similarity_type=configs['similarity_type'].name,
        word_embedding_type=configs['word_embedding_type'].name
    )

    table_columns = authors
    table_data = all_combinations_mean_pred

    __save_authors_table_image(table_filename, table_title, table_data, table_columns)

    # CONFIGURING CSV FILE
    base_filename = "prediction-similarity-values-all-combinations.csv"
    path_file = os.path.join(helper.get_results_path_directory_by_dataset(dataset_type), base_filename)
    mode_file = 'a' if os.path.exists(path_file) else 'w'
    has_header = mode_file == 'w'

    csv_columns = [
        'date', 'neural_network_type',
        'similarity_type', 'embedding_type',
        'max_seq_length', 'author1', 'author2', 'all_combination_mean_prediction'
    ]

    rows_data = []
    for i in range(len(authors_prod_combinations)):
        author1 = authors_prod_combinations[i][0]
        author2 = authors_prod_combinations[i][1]
        mean_pred = all_combinations_mean_pred[i]

        rows_data.append([
            configs['date'].strftime("%d/%m/%Y %H:%M:%S"), configs['neural_network_type'].name,
            configs['similarity_type'].name, configs['word_embedding_type'].name,
            configs['max_seq_length'], author1, author2, mean_pred
        ])

    dataframe = pd.DataFrame(rows_data, columns=csv_columns)
    dataframe.to_csv(path_file, index=False, mode=mode_file, header=has_header)


def save_correlation_metrics(df_prediction, predictions, configs):
    columns = [
        'date', 'neural_network_type',
        'similarity_type', 'embedding_type', 'max_seq_length',
        'pearson', 'spearman', 'mse'
    ]

    y_true = [float(row['label']) for index, row in df_prediction.iterrows()]
    y_pred = [float(pred[0]) for pred in predictions.tolist()]

    calculated_metrics = calculate_prediction_metrics(y_true, y_pred)

    row_data = [
        configs['date'].strftime("%d/%m/%Y %H:%M:%S"), configs['neural_network_type'].name,
        configs['similarity_type'].name, configs['word_embedding_type'].name, configs['max_seq_length'],
        calculated_metrics['pearson'], calculated_metrics['spearman'], calculated_metrics['mse']
    ]

    dataset_type = configs['dataset_type']
    base_filename = "prediction-metrics-results.csv"
    path_file = os.path.join(helper.get_results_path_directory_by_dataset(dataset_type), base_filename)
    mode_file = 'a' if os.path.exists(path_file) else 'w'
    has_header = mode_file == 'w'

    dataframe = pd.DataFrame([row_data], columns=columns)
    dataframe.to_csv(path_file, index=False, mode=mode_file, header=has_header)


def calculate_prediction_metrics(y_true, y_pred):
    pearson_val = scipy.stats.pearsonr(y_true, y_pred)[0]
    spearman_val = scipy.stats.spearmanr(y_true, y_pred)[0]
    mse_val = sklearn.metrics.mean_squared_error(y_true, y_pred)

    return {'pearson': pearson_val, 'spearman': spearman_val, 'mse': mse_val}


def __save_authors_table_image(table_filename, table_title, table_data, table_columns):
    # Save as a table image file
    fig, ax = plt.subplots()
    ax.set_axis_off()

    ax.table(
        cellText=table_data,
        rowLabels=table_columns,
        colLabels=table_columns,
        rowColours=['#99ddff'] * 10,
        colColours=['#99ddff'] * 10,
        cellLoc='center',
        loc='upper left'
    )

    ax.set_title(table_title, fontweight="bold")

    plt.tight_layout()
    plt.savefig(table_filename)
    plt.clf()


def __extract_authors_combinations_predictions(df_prediction, predictions_result):
    authors_predictions = {}
    predictions_list = __to_predictions_list(predictions_result)

    for index, row in df_prediction.iterrows():
        author1 = row['author1']
        author2 = row['author2']

        authors_key = author1 + "-" + author2

        if authors_key not in authors_predictions:
            authors_predictions[authors_key] = {
                'author1': author1,
                'author2': author2,
                'y_true': [],
                'y_pred': []
            }

        dic_pred = authors_predictions[authors_key]
        dic_pred['y_true'].append(float(row['label']))
        dic_pred['y_pred'].append(float(predictions_list[index]))

    return authors_predictions


def __extract_authors_phrases_vectors(df_prediction):
    # Phrases as a indices of vectors
    authors_phrases = {}

    for index, row in df_prediction.iterrows():
        author1 = row['author1']
        author2 = row['author2']

        for author in [author1, author2]:
            if author not in authors_phrases:
                authors_phrases[author] = []

        phrase_author1 = df_prediction['phrase1_n']
        phrase_author2 = df_prediction['phrase2_n']
        authors_phrases[author1].append(phrase_author1)
        authors_phrases[author2].append(phrase_author2)

    return authors_phrases


def __combination_between_same_authors(author_indices_vectors_phrases):
    left_side_input = []
    right_side_input = []
    start_index_inner_loop = 0

    for index_vector in author_indices_vectors_phrases:
        for i in range(start_index_inner_loop, len(author_indices_vectors_phrases), 1):
            left_side_input.append(index_vector)
            right_side_input.append(author_indices_vectors_phrases[i])

        start_index_inner_loop += 1

    return {'author1': left_side_input, 'author2': right_side_input}


def _combination_between_different_authors(author1_indices_vectors_phrases, author2_indices_vectors_phrases):
    author1_input = []
    author2_input = []

    for author1_phrases in author1_indices_vectors_phrases:
        for author2_phrases in author2_indices_vectors_phrases:
            author1_input.append(author1_phrases)
            author2_input.append(author2_phrases)

    return {'author1': author1_input, 'author2': author2_input}


def __to_predictions_list(predictions_result):
    return [pred[0] for pred in predictions_result.tolist()]
