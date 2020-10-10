# -*- coding: utf-8 -*-

"""
    Contains a set of functions to train the Neural Network.
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import src.core.helper as helper
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM
from src.models.ManhattanDistance import ManhattanDistance

matplotlib.use('Agg')


def load_training_dataframe(filename):
    train_dataframe = pd.read_csv(filename)

    for q in ['phrase1', 'phrase2']:
        train_dataframe[q + '_n'] = train_dataframe[q]

    return train_dataframe


def make_word2vec_embeddings(train_dataframe, embedding_dim=300, empty_w2v=False):
    train_df, embeddings = helper.make_w2v_embeddings(train_dataframe, embedding_dim=embedding_dim, empty_w2v=empty_w2v)
    train_dataframe = train_df

    return embeddings


def get_validation_size(train_dataframe, percent):
    return int(len(train_dataframe) * percent / 100)


def get_training_size(train_dataframe, validation_size):
    return len(train_dataframe) - validation_size


def split_data_train(train_dataframe):
    x_questions = train_dataframe[['phrase1_n', 'phrase2_n']]

    train_dataframe.label = pd.Categorical(train_dataframe.label)
    train_dataframe['label'] = train_dataframe.label.cat.codes
    y_labels = train_dataframe['label']

    return {'phrases': x_questions, 'labels': y_labels}


def define_train_and_validation_dataframe(x_phrases, y_labels, validation_size, max_seq_length):
    x_train, x_validation, y_train, y_validation = train_test_split(x_phrases, y_labels, test_size=validation_size)

    # Zero padding and splitting in two parts (left and right)
    x_train = helper.split_and_zero_padding(x_train, max_seq_length)
    x_validation = helper.split_and_zero_padding(x_validation, max_seq_length)

    # Convert labels to their numpy representations
    y_train = y_train.values
    y_validation = y_validation.values

    return {'train': {'x': x_train, 'y': y_train}, 'validation': {'x': x_validation, 'y': y_validation}}


def check_train_dataframe(x_train, y_train):
    assert x_train['left'].shape == x_train['right'].shape
    assert len(x_train['left']) == len(y_train)


def define_shared_model(embeddings, embedding_dim, max_seq_length, n_hidden):
    shared_model = Sequential()
    shared_model.add(Embedding(len(embeddings),
                     embedding_dim,
                     weights=[embeddings],
                     input_shape=(max_seq_length,),
                     trainable=False))

    # LSTM - Long Short Term Memory
    shared_model.add(LSTM(n_hidden))

    return shared_model


def define_manhattan_model(shared_model, max_seq_length):
    # The visible layer
    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,), dtype='int32')

    # Pack it all up into a Manhattan Distance model
    malstm_distance = ManhattanDistance()([shared_model(left_input), shared_model(right_input)])
    manhattan_model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])

    return manhattan_model


def compile_model(model, gpus):
    if gpus >= 2:
        # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
        model = tf.keras.utils.multi_gpu_model(model, gpus=gpus)

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])


def show_summary_model(model):
    model.summary()


def train_neural_network(model, data, batch_size, n_epoch):
    x_train = data['train']['x']
    y_train = data['train']['y']
    x_validation = data['validation']['x']
    y_validation = data['validation']['y']

    neural_network_trained = model.fit([x_train['left'], x_train['right']],
                                       y_train,
                                       batch_size=batch_size,
                                       epochs=n_epoch,
                                       validation_data=([x_validation['left'], x_validation['right']],
                                                        y_validation)
                                       )
    return neural_network_trained


def save_model(model, filename):
    model.save(filename)


def set_plot_accuracy(network_trained):
    # Plot accuracy
    plt.subplot(211)
    plt.plot(network_trained.history['accuracy'])
    plt.plot(network_trained.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')


def set_plot_loss(network_trained):
    # Plot loss
    plt.subplot(212)
    plt.plot(network_trained.history['loss'])
    plt.plot(network_trained.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')


def save_plot_graph(filename):
    plt.tight_layout(h_pad=1.0)
    plt.savefig(filename)


def show_plot_graph():
    # plt.tight_layout(h_pad=1.0)
    plt.show()


def clear_plot_graph():
    plt.clf()


def report_max_accuracy(network_trained):
    print(str(network_trained.history['val_accuracy'][-1])[:6] +
          "(max: " + str(max(network_trained.history['val_accuracy']))[:6] + ")")
