import time
import os
import tensorflow as tf
import tensorflow_addons as tfa
import src.user_interface.cli_input as ui
import src.user_interface.cli_output as uo
import src.core.helper as helper
import src.core.data_structuring as structuring
import src.core.training as training
import src.core.prediction as prediction
from datetime import datetime
from src.enums.DatasetType import DatasetType
from src.enums.WordEmbeddingType import WordEmbeddingType
from src.enums.NeuralNetworkType import  NeuralNetworkType
from src.enums.SimilarityMeasureType import SimilarityMeasureType
from src.enums.Stage import Stage


def main():
    stage = ui.choose_stage()
    uo.clear_screen()
    status = _execute_stage(stage)
    uo.break_lines(3)

    return status


def _execute_stage(stage):
    if stage == Stage.STRUCTURING:
        __execute_data_structuring()
        return 1
    elif stage == Stage.TRAINING:
        __execute_training()
        return 1
    elif stage == Stage.PREDICTION:
        __execute_prediction()
        return 1
    else:
        uo.show_leaving_message()
        return 0


def __execute_data_structuring():
    authors_dir_training = "./data/works/training"
    authors_dir_prediction = "./data/works/prediction"

    # User input variables
    n_sentences_training = ui.insert_number_sentences("Enter the number of sentences of each author to structure data TRAINING: ")
    uo.break_lines(1)
    n_sentences_prediction = ui.insert_number_sentences("Enter the number of sentences of each author to structure data PREDICTION: ")
    uo.break_lines(1)
    number_authors = ui.insert_number_of_authors()
    uo.break_lines(1)

    for dataset_type in list(DatasetType):
        # ----- TRAINING data structuring -----
        filename_training = helper.get_dataset_type_filename(dataset_type, "training-sentences-{dataset_type}.csv")
        print("Structuring and saving TRAINING {} file...".format(filename_training))

        # Extract the data from dataset
        authors = structuring.list_dir_authors(authors_dir_training, number_authors)
        dic_works = structuring.dic_works_by_authors(authors)
        dic_data_works = structuring.extract_works_sentence_data(dic_works, n_sentences_training, dataset_type, True)

        # Save TRAINING csv file with the extracted data
        structuring.validate_total_number_of_sentences(dic_data_works, n_sentences_training, number_authors, dataset_type, True)
        df_training = structuring.save_training_sentences_as_csv(dic_data_works, dataset_type, n_sentences_training)

        # Plot histogram from training dataset
        print("Plotting and saving sentences histogram...")
        dataset_training_name = os.path.join(helper.DATA_FILES_TRAINING_PATH, filename_training)
        training_dataframe = training.load_training_dataframe(dataset_training_name)
        training_dataframe = training_dataframe[:int((n_sentences_training * len(authors)) / 2)]
        structuring.plot_hist_length_dataframe(training_dataframe, dataset_type)

        # ----- PREDICTION data structuring -----
        filename_prediction = helper.get_dataset_type_filename(dataset_type, "prediction-sentences-{dataset_type}.csv")
        print("Structuring and saving PREDICTION {} file...".format(filename_training))

        # Extract the data from dataset
        authors = structuring.list_dir_authors(authors_dir_prediction, number_authors)
        dic_works = structuring.dic_works_by_authors(authors)
        dic_data_works = structuring.extract_works_sentence_data(dic_works, n_sentences_prediction, dataset_type, False)

        # Save PREDICTION csv file with the extracted data
        structuring.validate_total_number_of_sentences(dic_data_works, n_sentences_prediction, number_authors, dataset_type, False)
        df_prediction = structuring.save_prediction_sentences_as_csv(dic_data_works, dataset_type, n_sentences_prediction)

        for word_embedding_type in list(WordEmbeddingType):
            # Create index vectors and embeddings matrix
            word_embedding = helper.load_word_embedding(word_embedding_type)

            index_vector_message = "Creating and saving index vector of {} file for {} word embedding..."

            print(index_vector_message.format(filename_training, word_embedding_type.name))
            index_vector_training, vocabs_training = helper.create_index_vector(df_training, word_embedding)
            helper.save_index_vector(index_vector_training, dataset_type, word_embedding_type, training_process=True)

            print(index_vector_message.format(filename_prediction, word_embedding_type.name))
            index_vector_prediction, _ = helper.create_index_vector(df_prediction, word_embedding)
            helper.save_index_vector(index_vector_prediction, dataset_type, word_embedding_type, training_process=False)

            print("Creating and saving embedding matrix file for {} word embedding...".format(word_embedding_type.name))
            embedding_matrix = helper.create_embedding_matrix(word_embedding, vocabs_training, embedding_dim=300)
            helper.save_embedding_matrix(embedding_matrix, dataset_type, word_embedding_type)

            helper.delete_word_embedding(word_embedding)


def __execute_training():
    # Filename results
    hyperparameters_filename = "./data/training_variables.txt"
    model_save_filename = "./data/SiameseLSTM.h5"

    # Model variables (hyperparameters)
    gpus = 1

    # User input variables
    dataset_type = ui.insert_dataset_type()
    uo.break_lines(1)
    word_embedding_type = ui.insert_word_embedding()
    uo.break_lines(1)
    hyperparameters = ui.insert_hyperparameters_variables()

    # merge hyperparameters dicts based on neural network architecture
    if (hyperparameters['neural_network_type']) == NeuralNetworkType.LSTM:
        hyperparameters_lstm = {
            'gpus': gpus,
            'embedding_dim': 300,
            'max_seq_length': 30,
            'batch_size': 32 * gpus,
            'n_epochs': 20,
            'n_hidden': 128,
            'kernel_initializer': tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal',seed=1),
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None,
            'activation': "softsign",
            'recurrent_activation': "sigmoid",
            'dropout': 0.11,
            'recurrent_dropout': 0.32,
            'activation_layer': "selu",
            'activation_dense_layer': "sigmoid",
            'loss': tf.keras.losses.MeanSquaredError(),
            'optimizer': tf.keras.optimizers.Adamax(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adamax")
        }
        hyperparameters_lstm.update(hyperparameters)
        hyperparameters = hyperparameters_lstm
    else:
        hyperparameters_cnn = {
            'gpus': gpus,
            'embedding_dim': 300,
            'batch_size': 128 * gpus,
            'n_epochs': 20,
            'n_hidden': 300,
            'conv1d_filters': 250,
            'kernel_size': 5,
            'dense_units_relu': 250,
            'dense_units_sigmoid': 1,
            'activation_relu': "relu",
            'activation_sigmoid': "sigmoid",
            'dropout_rate': 0.3,
            'loss': tfa.losses.ContrastiveLoss(),
            'optimizer': tf.keras.optimizers.Adam(learning_rate=0.001)
        }
        hyperparameters_cnn.update(hyperparameters)
        hyperparameters = hyperparameters_cnn

    # Loading index vector and embedding matrix
    index_vector_filename = helper.get_index_vector_filename(dataset_type, word_embedding_type, training_process=True)
    embedding_matrix_filename = helper.get_embedding_matrix_filename(dataset_type, word_embedding_type)

    training_dataframe = helper.load_index_vector_dataframe(index_vector_filename)
    embedding_matrix = helper.load_embedding_matrix(embedding_matrix_filename)

    # Data preparation and normalization
    validation_size = training.get_validation_size(training_dataframe, hyperparameters['percent_validation'])
    training_size = training.get_training_size(training_dataframe, validation_size)
    splited_data_training = training.split_data_train(training_dataframe)
    normalized_dataframe = training.define_train_and_validation_dataframe(splited_data_training['phrases'],
                                                                          splited_data_training['labels'],
                                                                          validation_size,
                                                                          hyperparameters['max_seq_length'])

    # Creating the model based on a similarity function/measure
    shared_model = training.define_shared_model(embedding_matrix, hyperparameters)
    # training.show_summary_model(shared_model)

    model = training.define_model(shared_model, hyperparameters)
    training.compile_model(model, hyperparameters)
    training.show_summary_model(model)

    # Training the neural network based on model
    training_start_time = time.time()
    training_history = training.train_neural_network(model, normalized_dataframe, hyperparameters)
    training_end_time = time.time()
    uo.show_training_finished_message(hyperparameters['n_epochs'], training_start_time, training_end_time)

    training.save_model(model, model_save_filename)

    # Results
    training.set_plot_accuracy(training_history)
    training.set_plot_loss(training_history)
    graph_save_filename = (helper.get_results_path_directory_by_dataset(dataset_type) + "/history-graph-{date_now}-{word_embedding}-{network_type}-{similarity_type}-{percent_training}-{percent_validation}-{epochs}-{max_seq_length}.png").format(
        date_now=datetime.now().strftime("%d_%m_%Y-%H_%M_%S"),
        word_embedding=word_embedding_type.name,
        network_type=hyperparameters['neural_network_type'].name,
        similarity_type=hyperparameters['similarity_measure_type'].name,
        percent_training=int(training.get_percent_training_size(training_dataframe, training_size)),
        percent_validation=int(training.get_percent_validation_size(training_dataframe, validation_size)),
        epochs=hyperparameters['n_epochs'],
        max_seq_length=hyperparameters['max_seq_length']
    )
    training.save_plot_graph(graph_save_filename)
    training.clear_plot_graph()
    # training.show_plot_graph()
    training.report_max_accuracy(training_history)
    # training.report_size_data(training_dataframe, training_size, validation_size)

    # Save config file
    training.save_model_variables_file(hyperparameters_filename, hyperparameters)


def __execute_prediction():
    # Saved model trained
    hyperparameters_filename = "./data/training_variables.txt"
    model_saved_filename = "./data/SiameseLSTM.h5"
    table_title = "Índices de Similaridade ({network_type} - {similarity_type} - {word_embedding_type})"

    # Model variables
    hyperparameters = training.get_hyperparameters(hyperparameters_filename)

    # User input variables
    dataset_type = ui.insert_dataset_type()
    uo.break_lines(1)
    word_embedding_type = ui.insert_word_embedding()
    uo.break_lines(1)

    # Loading index vector and embedding matrix
    index_vector_filename = helper.get_index_vector_filename(dataset_type, word_embedding_type, training_process=False)
    prediction_dataframe = helper.load_index_vector_dataframe(index_vector_filename)

    # Data preparation
    test_normalized_dataframe = prediction.define_prediction_dataframe(prediction_dataframe, hyperparameters["max_seq_length"])

    # Loading the model trained
    test_model = prediction.load_model(model_saved_filename)
    prediction.show_summary_model(test_model)

    # Predict data from the model trained
    prediction_result = prediction.predict_neural_network(test_model, test_normalized_dataframe)

    # Results
    table_filename = helper.get_results_path_directory_by_dataset(dataset_type) + "/similarity-values-{network_type}-{similarity_type}-{word_embedding_type}.png"

    # prediction.save_prediction_result(
    #     prediction_result,
    #     50,
    #     table_title.format(
    #         network_type=hyperparameters["neural_network_type"],
    #         similarity_type=hyperparameters["similarity_measure_type"],
    #         word_embedding_type=word_embedding_type.name
    #     ),
    #     table_filename.format(
    #         network_type=hyperparameters["neural_network_type"],
    #         similarity_type=hyperparameters["similarity_measure_type"],
    #         word_embedding_type=word_embedding_type.name
    #     )
    # )


if __name__ == '__main__':
    status_main = 1
    while status_main == 1:
        status_main = main()
