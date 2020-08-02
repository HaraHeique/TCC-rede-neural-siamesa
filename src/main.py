import time
import src.user_interface.cli_input as ui
import src.user_interface.cli_output as uo
import src.core.training as training
import src.core.prediction as prediction
from src.enums.Stage import Stage


def main():
    uo.clear_screen()
    stage = ui.choose_stage()
    status = _execute_stage(stage)

    return status


def _execute_stage(stage):
    if stage == Stage.TRAINING:
        _execute_training()
        return 1
    elif stage == Stage.PREDICTION:
        _execute_prediction()
        return 1
    else:
        return 0


def _execute_training():
    # Filename results
    model_save_filename = "/data/SiameseLSTM.h5"
    graph_save_filename = "/results/history-graph.png"

    # Model variables
    max_seq_length = 35
    embedding_dim = 300
    gpus = 1
    batch_size = 128 * gpus
    n_epoch = 50
    n_hidden = 50

    # User input variables
    filename = ui.insert_training_filename()
    percent_validation = ui.insert_percent_validation()

    training_dataframe = training.load_training_dataframe(filename)
    embeddings_matrix = training.make_word2vec_embeddings(training_dataframe, embedding_dim)

    validation_size = training.get_validation_size(training_dataframe, percent_validation)
    training_size = training.get_training_size(training_dataframe, validation_size)

    splited_data_training = training.split_data_train(training_dataframe)
    normalized_dataframe = training.define_train_and_validation_dataframe(splited_data_training['questions'],
                                                                          splited_data_training['labels'],
                                                                          validation_size,
                                                                          max_seq_length)

    shared_model = training.define_shared_model(embeddings_matrix, embedding_dim, max_seq_length, n_hidden)
    training.show_summary_model(shared_model)

    manhattan_model = training.define_manhattan_model(shared_model, max_seq_length)
    training.compile_model(manhattan_model, gpus)
    training.show_summary_model(manhattan_model)

    training_start_time = time.time()
    manhattan_model_trained = training.train_neural_network(manhattan_model, normalized_dataframe, batch_size, n_epoch)
    training_end_time = time.time()
    uo.training_finished_message(n_epoch, training_start_time, training_end_time)

    training.save_model(manhattan_model_trained, model_save_filename)

    training.set_plot_accuracy(manhattan_model_trained)
    training.set_plot_loss(manhattan_model_trained)
    training.save_plot_graph(graph_save_filename)
    training.show_plot_graph()
    training.report_max_accuracy(manhattan_model_trained)


def _execute_prediction():
    # Saved model trained
    model_saved_filename = "/data/SiameseLSTM.h5"

    # Model variables
    max_seq_length = 35
    embedding_dim = 300

    # User input variables
    filename = ui.insert_prediction_filename()

    prediction_dataframe = prediction.load_prediction_dataframe(filename)
    embeddings_matrix = prediction.make_word2vec_embeddings(prediction_dataframe, embedding_dim)

    test_normalized_dataframe = prediction.define_prediction_dataframe(prediction_dataframe, max_seq_length)

    test_model = prediction.load_manhattan_model(model_saved_filename)
    prediction.show_summary_model(test_model)

    prediction.show_prediction_model(test_model, test_normalized_dataframe)


if __name__ == '__main__':
    status_main = 1
    while status_main == 1:
        status_main = main()
