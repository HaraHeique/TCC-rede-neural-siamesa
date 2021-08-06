# -*- coding: utf-8 -*-

"""
    Contains a set of functions to predict the Neural Network.
"""

import os
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


def save_prediction_metrics_by_author_combination(predictions, n_pairs, configs):
    dataset_type = configs['dataset_type']

    # LÓGICA REUTILIZÁVEL PARA CRIAR OS RESULTADOS DINÂMICOS PARA N AUTORES, POIS LÓGICA ATUAL É ESTÁTICA
    # authors_predictions = {}
    # predictions_list = [pred[0] for pred in predictions.tolist()]
    #
    # for index, row in df_prediction.iterrows():
    #     author1 = row['author1']
    #     author2 = row['author2']
    #
    #     authors_key = author1 + "-" + author2
    #
    #     if authors_key not in authors_predictions:
    #         authors_predictions[authors_key] = {
    #             'author1': author1,
    #             'author2': author2,
    #             'y_true': [],
    #             'y_pred': []
    #         }
    #
    #     dic_pred = authors_predictions[authors_key]
    #     dic_pred['y_true'].append(float(row['label']))
    #     dic_pred['y_pred'].append(float(predictions_list[index]))

    # CONFIGURING TABLE
    table_title = "Índices de Similaridade \n({network_type} - {similarity_type} - {word_embedding_type})".format(
        network_type=configs['neural_network_type'].name,
        similarity_type=configs['similarity_type'].name,
        word_embedding_type=configs['word_embedding_type'].name
    )

    table_filename = helper.get_results_path_directory_by_dataset(dataset_type) + "/prediction-similarity-values-{date}-{network_type}-{similarity_type}-{word_embedding_type}.png"
    table_filename = table_filename.format(
        date=configs['date'].strftime("%d_%m_%Y-%H_%M_%S"),
        network_type=configs['neural_network_type'].name,
        similarity_type=configs['similarity_type'].name,
        word_embedding_type=configs['word_embedding_type'].name
    )

    # Extracting data
    data_list = [pred[0] for pred in predictions.tolist()]
    mean_list = [(sum(data_list[i:i + n_pairs]) / n_pairs) for i in range(0, len(data_list), n_pairs) if
                 (i + n_pairs) <= len(data_list)]

    # Structuring data
    dic_data = {
        'faulkner-faulkner': "{:.2f}".format(mean_list[0]),
        'hemingway-hemingway': "{:.2f}".format(mean_list[1]),
        'roth-roth': "{:.2f}".format(mean_list[2]),
        'faulkner-hemingway': "{:.2f}".format(mean_list[3]),
        'faulkner-roth': "{:.2f}".format(mean_list[4]),
        'hemingway-roth': "{:.2f}".format(mean_list[5]),
    }

    table_columns = ["Faulkner", "Hemingway", "Roth"]
    table_data = [
        [dic_data['faulkner-faulkner'], dic_data['faulkner-hemingway'], dic_data['faulkner-roth']],
        [dic_data['faulkner-hemingway'], dic_data['hemingway-hemingway'], dic_data['hemingway-roth']],
        [dic_data['faulkner-roth'], dic_data['hemingway-roth'], dic_data['roth-roth']]
    ]

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
    for author_combination, pred in dic_data.items():
        authors_splited = author_combination.split('-')

        rows_data.append([
            configs['date'].strftime("%d/%m/%Y %H:%M:%S"), configs['neural_network_type'].name,
            configs['similarity_type'].name, configs['word_embedding_type'].name,
            configs['max_seq_length'], authors_splited[0], authors_splited[1], pred
        ])

    dataframe = pd.DataFrame(rows_data, columns=columns)
    dataframe.to_csv(path_file, index=False, mode=mode_file, header=has_header)


def save_prediction_metrics_global(df_prediction, predictions, configs):
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
