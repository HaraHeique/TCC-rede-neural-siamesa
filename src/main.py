import time
import os
import src.user_interface.cli_input as ui
import src.user_interface.cli_output as uo
import src.core.helper as helper
import src.core.data_structuring as structuring
import src.core.training as training
import src.core.prediction as prediction
from src.enums.DatasetType import DatasetType
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
    quantity_sentences_filename = "quantity-sentences-by-works-{}.csv"
    authors_dir_training = "./data/works/training"
    authors_dir_prediction = "./data/works/prediction"

    # User input variables
    # dataset_type = ui.insert_dataset_type()
    # uo.break_lines(1)
    n_sentences_training = ui.insert_number_sentences("Enter the number of sentences of each author to structure data TRAINING: ")
    uo.break_lines(1)
    n_sentences_prediction = ui.insert_number_sentences("Enter the number of sentences of each author to structure data PREDICTION: ")
    uo.break_lines(1)

    for dataset_type in list(DatasetType):
        # ----- TRAINING data structuring -----
        base_filename_training = "training-sentences-{dataset_type}.csv"
        print("Structuring and saving TRAINING {} file...".format(helper.get_dataset_type_filename(dataset_type, base_filename_training)))

        # Extract the data from dataset
        authors = structuring.list_dir_authors(authors_dir_training)
        dic_works = structuring.dic_works_by_authors(authors)
        dic_data_works = structuring.extract_works_sentence_data(dic_works, n_sentences_training, dataset_type, quantity_sentences_filename.format("training"))

        # Save TRAINING csv file with the extracted data
        structuring.save_training_sentences_as_csv(dic_data_works, dataset_type, n_sentences_training, 4)

        # Plot histogram from training dataset
        print("Plotting and saving sentences histogram...")
        dataset_training_name = os.path.join(helper.DATA_FILES_TRAINING_PATH, helper.get_dataset_type_filename(dataset_type, base_filename_training))
        training_dataframe = training.load_training_dataframe(dataset_training_name)
        training_dataframe = training_dataframe[:int((n_sentences_training * len(authors)) / 2)]
        structuring.plot_hist_length_dataframe(training_dataframe, dataset_type)

        # ----- PREDICTION data structuring -----
        base_filename_prediction = "prediction-sentences-{dataset_type}.csv"
        print("Structuring and saving PREDICTION {} file...".format(helper.get_dataset_type_filename(dataset_type, base_filename_prediction)))

        # Extract the data from dataset
        authors = structuring.list_dir_authors(authors_dir_prediction)
        dic_works = structuring.dic_works_by_authors(authors)
        dic_data_works = structuring.extract_works_sentence_data(dic_works, n_sentences_prediction, dataset_type, quantity_sentences_filename.format("prediction"))

        # Save PREDICTION csv file with the extracted data
        structuring.save_prediction_sentences_as_csv(dic_data_works, dataset_type, n_sentences_prediction, 4)


def __execute_training():
    # Filename results
    base_filename_training = "training-sentences-{dataset_type}.csv"
    hyperparameters_filename = "./data/training_variables.txt"
    model_save_filename = "./data/SiameseLSTM.h5"

    # Model variables (hyperparameters)
    gpus = 1

    hyperparameters = {
        'embedding_dim': 300,
        'gpus': gpus,
        'batch_size': 128 * gpus,
        'n_epochs': 50,
        'n_hidden': 50,
        'conv1d_filters': 250,
        'kernel_size': 5,
        'dense_units_relu': 250,
        'dense_units_sigmoid': 1,
        'activation_relu': "relu",
        'activation_sigmoid': "sigmoid",
        'dropout_rate': 0.3,
        'loss': "mean_squared_error",
        'optimizer': "adam"
    }

    # User input variables
    # filename = ui.insert_training_filename()
    dataset_type = ui.insert_dataset_type()
    training_filename = helper.get_dataset_type_path_filename(dataset_type, base_filename_training)
    uo.break_lines(1)
    word_embedding_type = ui.insert_word_embedding()
    word_embedding_file = helper.get_word_embedding_path_filename(word_embedding_type)
    uo.break_lines(1)
    ui.insert_hyperparameters_variables(hyperparameters)

    # Data loading
    training_dataframe = training.load_training_dataframe(training_filename)

    # Data pre-processing and creating embedding matrix
    training_dataframe, embedding_matrix = training.make_word_embeddings(word_embedding_file, training_dataframe, hyperparameters['embedding_dim'])

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
    model_trained = training.train_neural_network(model, normalized_dataframe, hyperparameters)
    training_end_time = time.time()
    uo.show_training_finished_message(hyperparameters['n_epochs'], training_start_time, training_end_time)

    training.save_model(model, model_save_filename)

    # Results
    training.set_plot_accuracy(model_trained)
    training.set_plot_loss(model_trained)
    graph_save_filename = (helper.get_results_path_directory_by_dataset(dataset_type) + "/history-graph-{word_embedding}-{network_type}-{similarity_type}-{percent_training}-{percent_validation}-{epochs}-{max_seq_length}.png").format(
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
    training.report_max_accuracy(model_trained)
    # training.report_size_data(training_dataframe, training_size, validation_size)

    # Save config file
    training.save_model_variables_file(hyperparameters_filename, hyperparameters)


def __execute_prediction():
    # Saved model trained
    base_filename_prediction = "prediction-sentences-{dataset_type}.csv"
    hyperparameters_filename = "./data/training_variables.txt"
    model_saved_filename = "./data/SiameseLSTM.h5"
    table_title = "Índices de Similaridade ({network_type} - {similarity_type} - {word_embedding_type})"

    # Model variables
    hyperparameters = training.get_hyperparameters(hyperparameters_filename)

    # User input variables
    # filename = ui.insert_prediction_filename()
    dataset_type = ui.insert_dataset_type()
    prediction_filename = helper.get_dataset_type_path_filename(dataset_type, base_filename_prediction, training_process=False)
    uo.break_lines(1)
    word_embedding_type = ui.insert_word_embedding()
    word_embedding_file = helper.get_word_embedding_path_filename(word_embedding_type)
    uo.break_lines(1)

    # Data loading
    prediction_dataframe = prediction.load_prediction_dataframe(prediction_filename)

    # Data pre-processing and creating embedding matrix
    prediction_dataframe, embeddings_matrix = prediction.make_word_embeddings(word_embedding_file, prediction_dataframe, hyperparameters["embedding_dim"])

    # Data preparation
    test_normalized_dataframe = prediction.define_prediction_dataframe(prediction_dataframe, hyperparameters["max_seq_length"])

    # Loading the model trained
    test_model = prediction.load_model(model_saved_filename)
    prediction.show_summary_model(test_model)

    # Predict data from the model trained
    prediction_result = prediction.predict_neural_network(test_model, test_normalized_dataframe)

    # Results
    table_filename = helper.get_results_path_directory_by_dataset(dataset_type) + "/similarity-values-{network_type}-{similarity_type}-{word_embedding_type}.png"

    prediction.save_prediction_result(
        prediction_result,
        50,
        table_title.format(
            network_type=hyperparameters["neural_network_type"],
            similarity_type=hyperparameters["similarity_measure_type"],
            word_embedding_type=word_embedding_type.name
        ),
        table_filename.format(
            network_type=hyperparameters["neural_network_type"],
            similarity_type=hyperparameters["similarity_measure_type"],
            word_embedding_type=word_embedding_type.name
        )
    )


if __name__ == '__main__':
    status_main = 1
    while status_main == 1:
        status_main = main()
