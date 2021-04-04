import time
import src.user_interface.cli_input as ui
import src.user_interface.cli_output as uo
import src.core.helper as helper
import src.core.data_structuring as structuring
import src.core.training as training
import src.core.prediction as prediction
from src.enums.Stage import Stage
from src.enums.SimilarityMeasureType import SimilarityMeasureType
from src.enums.NeuralNetworkType import NeuralNetworkType


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
    # ----- TRAINING data structuring -----

    authors_dir_training = "./data/works/training"
    filename = "./data/training/training-{n_sentences}-sentences.csv"

    # User input variables
    n_sentences_training = ui.insert_number_sentences("Enter the number of sentences of each author to structure data TRAINING: ")
    print("Structuring and saving TRAINING csv file...")

    # Extract the data from dataset
    authors = structuring.list_dir_authors(authors_dir_training)
    dic_works = structuring.dic_works_by_authors(authors)
    dic_data_works = structuring.extract_works_sentence_data(dic_works, n_sentences_training)

    # Save TRAINING csv file with the extracted data
    structuring.save_training_sentences_as_csv(dic_data_works, n_sentences_training)

    # Plot histogram from training dataset
    print("Plotting and saving sentences histogram...")
    distribution_save_filename = "./results/sentences-distribution.png"
    training_dataframe = training.load_training_dataframe(
        filename.format(n_sentences=n_sentences_training * len(authors))
    )
    helper.plot_hist_length_dataframe(training_dataframe, distribution_save_filename)

    # ----- PREDICTION data structuring -----
    authors_dir_prediction = "./data/works/prediction"

    # User input variables
    n_sentences_prediction = ui.insert_number_sentences("Enter the number of sentences of each author to structure data PREDICTION: ")
    print("Structuring and saving PREDICTION csv file...")

    # Extract the data from dataset
    authors = structuring.list_dir_authors(authors_dir_prediction)
    dic_works = structuring.dic_works_by_authors(authors)
    dic_data_works = structuring.extract_works_sentence_data(dic_works, n_sentences_prediction)

    # Save PREDICTION csv file with the extracted data
    structuring.save_prediction_sentences_as_csv(dic_data_works, n_sentences_prediction)


def __execute_training():
    # Filename results
    model_save_filename = "./data/SiameseLSTM.h5"
    graph_save_filename = "./results/history-graph-{network_type}-{similarity_type}-{percent_training}-{percent_validation}-{epochs}-{max_seq_length}.png"

    # Model variables
    max_seq_length = 5
    embedding_dim = 300
    gpus = 1
    batch_size = 128 * gpus
    n_epochs = 50
    n_hidden = 50

    # User input variables
    filename = ui.insert_training_filename()
    uo.break_lines(1)
    neural_network_type = ui.insert_neural_network_type()
    uo.break_lines(1)
    similarity_measure_type = ui.insert_similarity_measure_type()
    uo.break_lines(1)
    percent_validation = ui.insert_percent_validation()
    uo.break_lines(1)
    n_epochs = ui.insert_number_epochs()
    uo.break_lines(1)
    max_seq_length = ui.insert_max_seq_length()
    uo.break_lines(1)

    # Data loading
    training_dataframe = training.load_training_dataframe(filename)

    # Data pre-processing and creating embedding matrix
    embedding_matrix = training.make_word2vec_embeddings(training_dataframe, embedding_dim)

    # Data preparation and normalization
    validation_size = training.get_validation_size(training_dataframe, percent_validation)
    training_size = training.get_training_size(training_dataframe, validation_size)
    splited_data_training = training.split_data_train(training_dataframe)
    normalized_dataframe = training.define_train_and_validation_dataframe(splited_data_training['phrases'],
                                                                          splited_data_training['labels'],
                                                                          validation_size,
                                                                          max_seq_length)

    # Creating the model based on a similarity function/measure
    shared_model = training.define_shared_model(embedding_matrix, embedding_dim, max_seq_length, n_hidden,
                                                neural_network_type)
    # training.show_summary_model(shared_model)

    model = training.define_model(shared_model, max_seq_length, similarity_measure_type)
    training.compile_model(model, gpus)
    training.show_summary_model(model)

    # Training the neural network based on model
    training_start_time = time.time()
    manhattan_model_trained = training.train_neural_network(model, normalized_dataframe, batch_size, n_epochs)
    training_end_time = time.time()
    uo.show_training_finished_message(n_epochs, training_start_time, training_end_time)

    training.save_model(model, model_save_filename)

    # Results
    training.set_plot_accuracy(manhattan_model_trained)
    training.set_plot_loss(manhattan_model_trained)
    graph_save_filename = graph_save_filename.format(
        network_type=neural_network_type.name,
        similarity_type=similarity_measure_type.name,
        percent_training=int(training.get_percent_training_size(training_dataframe, training_size)),
        percent_validation=int(training.get_percent_validation_size(training_dataframe, validation_size)),
        epochs=n_epochs,
        max_seq_length=max_seq_length
    )
    training.save_plot_graph(graph_save_filename)
    training.clear_plot_graph()
    # training.show_plot_graph()
    training.report_max_accuracy(manhattan_model_trained)
    # training.report_size_data(training_dataframe, training_size, validation_size)


def __execute_prediction():
    # Saved model trained
    model_saved_filename = "./data/SiameseLSTM.h5"
    table_title = "Índices de Similaridade ({network_type} - {similarity_type})"
    table_filename = "./results/similarity-values-{network_type}-{similarity_type}.png"

    # Model variables
    max_seq_length = 35
    embedding_dim = 300

    # User input variables
    filename = ui.insert_prediction_filename()
    uo.break_lines(1)
    max_seq_length = ui.insert_max_seq_length()
    uo.break_lines(1)

    # Data loading
    prediction_dataframe = prediction.load_prediction_dataframe(filename)

    # Data pre-processing and creating embedding matrix
    embeddings_matrix = prediction.make_word2vec_embeddings(prediction_dataframe, embedding_dim)

    # Data preparation
    test_normalized_dataframe = prediction.define_prediction_dataframe(prediction_dataframe, max_seq_length)

    # Loading the model trained
    test_model = prediction.load_model(model_saved_filename)
    prediction.show_summary_model(test_model)

    # Predict data from the model trained
    prediction_result = prediction.predict_neural_network(test_model, test_normalized_dataframe)

    # Results
    network_name = NeuralNetworkType.LSTM.name
    similarity_name = SimilarityMeasureType.MANHATTAN.name

    prediction.save_prediction_result(
        prediction_result,
        100,
        table_title.format(
            network_type=network_name,
            similarity_type=similarity_name
        ),
        table_filename.format(
            network_type=network_name,
            similarity_type=similarity_name
        )
    )


if __name__ == '__main__':
    status_main = 1
    while status_main == 1:
        status_main = main()
