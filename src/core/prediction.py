# -*- coding: utf-8 -*-

"""
    Contains a set of functions to predict the Neural Network.
"""

import pandas as pd
import tensorflow as tf
import src.core.helper as helper
from src.models.ManhattanDistance import ManhattanDistance


def load_prediction_dataframe(filename):
    # Load training set
    prediction_dataframe = pd.read_csv(filename)
    for q in ['question1', 'question2']:
        prediction_dataframe[q + '_n'] = prediction_dataframe[q]

    return prediction_dataframe


def make_word2vec_embeddings(prediction_dataframe, embedding_dim=300, empty_w2v=False):
    prediction_dataframe, embeddings = helper.make_w2v_embeddings(prediction_dataframe,
                                                                  embedding_dim=embedding_dim,
                                                                  empty_w2v=empty_w2v)
    return embeddings


def define_prediction_dataframe(prediction_dataframe, max_seq_length):
    # Split to dicts and append zero padding.
    x_prediction = helper.split_and_zero_padding(prediction_dataframe, max_seq_length)

    return x_prediction


def check_prediction_dataframe(x_prediction):
    # Make sure everything is ok
    assert x_prediction['left'].shape == x_prediction['right'].shape


def load_manhattan_model(filename):
    model = tf.keras.models.load_model(filename, custom_objects={'ManhattanDistance': ManhattanDistance})

    return model


def show_summary_model(model):
    model.summary()


def show_prediction_model(model, x_prediction):
    prediction = model.predict([x_prediction['left'], x_prediction['right']])
    print(prediction)
