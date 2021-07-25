# -*- coding: utf-8 -*-

"""
    Contains a set of functions to train the Neural Network.
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import src.core.helper as helper
import src.core.similarity_measure as similarity_measure

from keras.layers.pooling import GlobalMaxPool1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM, Lambda, Conv1D, Dense, Dropout, Activation, MaxPooling1D, Flatten, Bidirectional
from src.enums.SimilarityMeasureType import SimilarityMeasureType
from src.enums.NeuralNetworkType import NeuralNetworkType

matplotlib.use('Agg')


def load_training_dataframe(filename):
    train_dataframe = pd.read_csv(filename)

    for q in ['phrase1', 'phrase2']:
        train_dataframe[q + '_n'] = train_dataframe[q]

    return train_dataframe


def find_max_seq_length(train_dataframe):
    return helper.find_max_seq_length(train_dataframe)


def get_validation_size(train_dataframe, percent):
    return int(len(train_dataframe) * percent / 100)


def get_training_size(train_dataframe, validation_size):
    return len(train_dataframe) - validation_size


def get_percent_validation_size(train_dataframe, validation_size):
    return (validation_size / len(train_dataframe)) * 100


def get_percent_training_size(train_dataframe, training_size):
    return (training_size / len(train_dataframe)) * 100


def split_data_train(train_dataframe):
    x_phrases = train_dataframe[['phrase1_n', 'phrase2_n']]

    train_dataframe.label = pd.Categorical(train_dataframe.label)
    train_dataframe['label'] = train_dataframe.label.cat.codes
    y_labels = train_dataframe['label']

    return {'phrases': x_phrases, 'labels': y_labels}


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


def define_shared_model(embeddings, hyperparameters):
    shared_model = Sequential()
    shared_model.add(Embedding(len(embeddings),
                               hyperparameters['embedding_dim'],
                               weights=[embeddings],
                               input_shape=(hyperparameters['max_seq_length'],),
                               trainable=False))

    np.random.seed(1)

    if hyperparameters['neural_network_type'] == NeuralNetworkType.CNN:
        # CNN - Convolutional Neural Network
        # shared_model.add(Conv1D(hyperparameters['conv1d_filters'], kernel_size=hyperparameters['kernel_size'], activation=hyperparameters['activation_relu']))
        # shared_model.add(GlobalMaxPool1D())
        # shared_model.add(Dense(hyperparameters['dense_units_relu'], activation=hyperparameters['activation_relu']))
        # shared_model.add(Dropout(hyperparameters['dropout_rate']))
        # shared_model.add(Dense(hyperparameters['dense_units_sigmoid'], activation=hyperparameters['activation_sigmoid']))
        shared_model.add(Conv1D(filters=300, kernel_size=5, activation='elu', use_bias=True, kernel_initializer=tf.keras.initializers.VarianceScaling()))
        shared_model.add(MaxPooling1D(pool_size=3))
        shared_model.add(Flatten())
        shared_model.add(Dense(300, activation='elu', kernel_initializer=tf.initializers.VarianceScaling(), bias_initializer=tf.initializers.VarianceScaling()))
        shared_model.add(Dense(1, activation='sigmoid'))
        shared_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    else:
        # LSTM - Long Short Term Memory
        shared_model.add(Dropout(hyperparameters['dropout']))
        shared_model.add(Bidirectional(LSTM(
            hyperparameters['n_hidden'],
            kernel_initializer=hyperparameters['kernel_initializer'],
            activation=hyperparameters['activation'],
            recurrent_activation=hyperparameters['recurrent_activation'],
            dropout=0.0,
            recurrent_dropout=hyperparameters['recurrent_dropout'],
            implementation=1
        )))
        shared_model.add(Activation(hyperparameters['activation_layer']))
        shared_model.add(Dense(1, activation=hyperparameters['activation_dense_layer']))
        # shared_model.add(LSTM(hyperparameters['n_hidden']))

    return shared_model


def define_model(shared_model, hyperparameters):
    similarity_type = hyperparameters['similarity_measure_type']
    max_seq_length = hyperparameters['max_seq_length']

    if similarity_type == SimilarityMeasureType.MANHATTAN:
        return define_manhattan_model(shared_model, max_seq_length)
    elif similarity_type == SimilarityMeasureType.COSINE:
        return define_cosine_model(shared_model, max_seq_length)
    elif similarity_type == SimilarityMeasureType.EUCLIDEAN:
        return define_euclidean_model(shared_model, max_seq_length)
    elif similarity_type == SimilarityMeasureType.JACCARD:
        return define_jaccard_model(shared_model, max_seq_length)
    else:
        return None


def define_manhattan_model(shared_model, max_seq_length):
    # The visible layer
    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,), dtype='int32')

    # Pack it all up into a Manhattan Distance model
    malstm_distance = similarity_measure.calculate_manhattan_distance(shared_model(left_input),
                                                                      shared_model(right_input))
    manhattan_model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])

    return manhattan_model


def define_cosine_model(shared_model, max_seq_length):
    left_input = Input(shape=(max_seq_length,))
    right_input = Input(shape=(max_seq_length,))

    cosine_distance = similarity_measure.calculate_cosine_distance(shared_model(left_input), shared_model(right_input))
    cosine_model = Model(inputs=[left_input, right_input], outputs=[cosine_distance])

    return cosine_model


def define_euclidean_model(shared_model, max_seq_length):
    left_input = Input(shape=(max_seq_length,))
    right_input = Input(shape=(max_seq_length,))

    euclidean_distance = Lambda(
        similarity_measure.calculate_euclidean_distance,
        output_shape=similarity_measure.dist_output_shape
    )([shared_model(left_input), shared_model(right_input)])
    euclidean_model = Model(inputs=[left_input, right_input], outputs=[euclidean_distance])

    return euclidean_model


def define_jaccard_model(shared_model, max_seq_length):
    left_input = Input(shape=(max_seq_length,))
    right_input = Input(shape=(max_seq_length,))

    jaccard_distance = Lambda(
        similarity_measure.calculate_jaccard_distance,
        output_shape=similarity_measure.dist_output_shape
    )([shared_model(left_input), shared_model(right_input)])
    jaccard_model = Model(inputs=[left_input, right_input], outputs=[jaccard_distance])

    return jaccard_model


def compile_model(model, hyperparameters):
    gpus = hyperparameters['gpus']

    if gpus >= 2:
        # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
        model = tf.keras.utils.multi_gpu_model(model, gpus=gpus)

    # model.compile(loss=tfa.losses.ContrastiveLoss(), optimizer=tf.keras.optimizers.RMSprop(), metrics=['accuracy'])
    # model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    # model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=hyperparameters['optimizer'], metrics=['accuracy'])
    model.compile(loss=hyperparameters['loss'], optimizer=hyperparameters['optimizer'], metrics=['accuracy'])


def show_summary_model(model):
    model.summary()


def train_neural_network(model, data, hyperparameters):
    x_train = data['train']['x']
    y_train = data['train']['y']
    x_validation = data['validation']['x']
    y_validation = data['validation']['y']

    neural_network_trained = model.fit([x_train['left'], x_train['right']],
                                       y_train,
                                       batch_size=hyperparameters['batch_size'],
                                       epochs=hyperparameters['n_epochs'],
                                       validation_data=([x_validation['left'], x_validation['right']],
                                                        y_validation)
                                       )
    return neural_network_trained


def save_model(model, filename):
    model.save(filename)


def set_plot_accuracy(training_history):
    # Plot accuracy
    plt.subplot(211)
    plt.plot(training_history.history['accuracy'])
    plt.plot(training_history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')


def set_plot_loss(training_history):
    # Plot loss
    plt.subplot(212)
    plt.plot(training_history.history['loss'])
    plt.plot(training_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')


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


def report_size_data(training_dataframe, training_size, validation_size):
    print("Size dataframe: {0} records\n"
          "Size training: {1} records\n"
          "Size validation: {2} records"
          .format(len(training_dataframe), training_size, validation_size))


def save_model_variables_file(filename, hyperparameters):
    content = """
        max_seq_length={max_seq_length}
        embedding_dim={embedding_dim}
        gpus={gpus}
        batch_size={batch_size}
        n_epochs={n_epochs}
        n_hidden={n_hidden}
        neural_network_type={neural_network_type}
        similarity_measure_type={similarity_measure_type}
        percent_validation={percent_validation}
    """.format(
        max_seq_length=hyperparameters["max_seq_length"],
        embedding_dim=hyperparameters["embedding_dim"],
        gpus=hyperparameters["gpus"],
        batch_size=hyperparameters["batch_size"],
        n_epochs=hyperparameters["n_epochs"],
        n_hidden=hyperparameters["n_hidden"],
        neural_network_type=hyperparameters["neural_network_type"].name,
        similarity_measure_type=hyperparameters["similarity_measure_type"].name,
        percent_validation=hyperparameters["percent_validation"]
    )

    content = content.replace(' ', '')

    try:
        f = open(filename, 'w')
        f.write(content)
        f.close()
    except Exception as err:
        print(err)


def get_hyperparameters(filename):
    try:
        f = open(filename, 'r')
        content = f.read()
        variables = {}
        SEPARATOR = '='

        for line in content.split('\n'):
            if SEPARATOR not in line:
                continue

            token = line.split(SEPARATOR)
            var_name = token[0]
            var_value = token[1]
            variables[var_name] = var_value if not var_value.isnumeric() else int(var_value)

        f.close()
    except Exception as err:
        print(err)
    else:
        return variables
