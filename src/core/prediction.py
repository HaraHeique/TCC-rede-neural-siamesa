# -*- coding: utf-8 -*-

"""
    Contains a set of functions to predict the Neural Network.
"""

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


def make_word_embeddings(word_embedding_filename, prediction_dataframe, embedding_dim=300, empty_w2v=False):
    prediction_dataframe, embeddings = helper.make_word_embeddings(word_embedding_filename,
                                                                   prediction_dataframe,
                                                                   embedding_dim=embedding_dim,
                                                                   empty_w2v=empty_w2v)
    return prediction_dataframe, embeddings


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


def save_prediction_result(prediction, n_pairs, title, filename):
    # Extracting data
    data_list = [data[0] for data in prediction]
    mean_list = [(sum(data_list[i:i+n_pairs]) / n_pairs) for i in range(0, len(data_list), n_pairs) if (i + n_pairs) <= len(data_list)]

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

    ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
