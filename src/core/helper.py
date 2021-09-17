# -*- coding: utf-8 -*-

"""
    Contains a set of reusable functions to help both training and prediction modules.
"""

import itertools
import re
import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from src.enums.DatasetType import DatasetType
from src.enums.WordEmbeddingType import WordEmbeddingType

SOURCE_PATH = os.path.dirname(os.path.abspath("src"))
# SOURCE_PATH = os.path.abspath("src")
DATA_FILES_PATH = SOURCE_PATH + "/data"
DATA_FILES_WORD_EMBEDDINGS_PATH = DATA_FILES_PATH + "/word_embeddings"
DATA_FILES_TRAINING_PATH = DATA_FILES_PATH + "/training"
DATA_FILES_PREDICTION_PATH = DATA_FILES_PATH + "/prediction"
DATA_FILES_INDEX_VECTORS_PATH = DATA_FILES_PATH + "/processed/index_vectors"
DATA_FILES_EMBEDDING_MATRICES_PATH = DATA_FILES_PATH + "/processed/embedding_matrices"
DATA_FILES_NETWORKS_MODELS_PATH = DATA_FILES_PATH + "/processed/networks_models"
DATA_FILES_RESULTS_PATH = SOURCE_PATH + "/results"


def get_dataset_type_filename(dataset_type, base_filename):
    if dataset_type == DatasetType.RAW:
        return base_filename.format(dataset_type="raw")
    elif dataset_type == DatasetType.WITHOUT_SW:
        return base_filename.format(dataset_type="sw")
    elif dataset_type == DatasetType.WITHOUT_SW_WITH_LEMMA:
        return base_filename.format(dataset_type="sw-lemmatization")


def get_dataset_type_path_filename(dataset_type, base_filename, training_process=True):
    file_path = DATA_FILES_TRAINING_PATH if training_process else DATA_FILES_PREDICTION_PATH

    if dataset_type == DatasetType.RAW:
        return os.path.join(file_path, get_dataset_type_filename(dataset_type, base_filename))
    elif dataset_type == DatasetType.WITHOUT_SW:
        return os.path.join(file_path, get_dataset_type_filename(dataset_type, base_filename))
    elif dataset_type == DatasetType.WITHOUT_SW_WITH_LEMMA:
        return os.path.join(file_path, get_dataset_type_filename(dataset_type, base_filename))


def get_results_path_directory_by_dataset(dataset_type):
    if dataset_type == DatasetType.RAW:
        return os.path.join(DATA_FILES_RESULTS_PATH, "raw")
    elif dataset_type == DatasetType.WITHOUT_SW:
        return os.path.join(DATA_FILES_RESULTS_PATH, "sw")
    elif dataset_type == DatasetType.WITHOUT_SW_WITH_LEMMA:
        return os.path.join(DATA_FILES_RESULTS_PATH, "sw_lemmatization")


def get_experiments_path_directory_by_dataset(dataset_type):
    return os.path.join(get_results_path_directory_by_dataset(dataset_type), "experiments")


def get_word_embedding_path_filename(word_embedding_type):
    word_embedding_files = {
        WordEmbeddingType.GLOVE_COMMON_CRAWL_UNCASED: "glove-CommonCrawl-uncased-42B-300d.txt",
        WordEmbeddingType.GLOVE_WIKIPEDIA_GIGAWORD: "glove-Wikipedia2014+Gigaword5-6B-300d.txt",
        WordEmbeddingType.WORD2VEC_WIKIPEDIA: "word2vec-EnglishWikipedia-300.bin",
        WordEmbeddingType.WORD2VEC_GOOGLE_NEWS: "word2vec-GoogleNews-300.bin.gz"
    }

    return os.path.join(DATA_FILES_WORD_EMBEDDINGS_PATH, word_embedding_files[word_embedding_type])


def get_index_vector_filename(dataset_type, word_embedding_type, training_process=True):
    base_filename = "{}-{}-{}.csv"
    dic_dataset_type = __get_dic_abbreviation_dataset_type()
    dic_word_embedding_type = __get_dic_abbreviation_word_embedding_type()
    process_name = "training" if training_process else "prediction"
    filename = base_filename.format(process_name, dic_dataset_type[dataset_type], dic_word_embedding_type[word_embedding_type])

    return os.path.join(DATA_FILES_INDEX_VECTORS_PATH, filename)


def get_embedding_matrix_filename(dataset_type, word_embedding_type):
    base_filename = "{}-{}.npy"
    dic_abbr_word_embedding_type = __get_dic_abbreviation_word_embedding_type()
    dic_abbr_dataset_type = __get_dic_abbreviation_dataset_type()
    filename = base_filename.format(
        dic_abbr_dataset_type[dataset_type],
        dic_abbr_word_embedding_type[word_embedding_type]
    )

    return os.path.join(DATA_FILES_EMBEDDING_MATRICES_PATH, filename)


def get_saved_model_filename(network_type, similarity_measure_type, dataset_type, word_embedding_type):
    base_filename = "Siamese-{}-{}-{}-{}.h5"
    dic_abbr_word_embedding_type = __get_dic_abbreviation_word_embedding_type()
    dic_abbr_dataset_type = __get_dic_abbreviation_dataset_type()

    filename = base_filename.format(
        network_type.name,
        similarity_measure_type.name,
        dic_abbr_dataset_type[dataset_type],
        dic_abbr_word_embedding_type[word_embedding_type]
    )

    return os.path.join(DATA_FILES_NETWORKS_MODELS_PATH, filename)


def create_index_vector(df_raw, word_embedding, vocabs=None):
    if vocabs is None:
        vocabs = {}

    vocabs_cnt = len(vocabs.keys())

    vocabs_not_w2v = {}
    vocabs_not_w2v_cnt = 0

    for index, row in df_raw.iterrows():
        # Print the number of embedded sentences.
        if index != 0 and index % 1000 == 0:
            print("{:,} sentences embedded.".format(index), flush=True)

        # Iterate through the text of both questions of the row
        for phrase in ['phrase1', 'phrase2']:

            # Question numbers representation
            q2n = []

            for word in __text_to_word_list(row[phrase]):
                # If a word is missing from word2vec model.
                if word not in word_embedding.vocab:
                    if word not in vocabs_not_w2v:
                        vocabs_not_w2v_cnt += 1
                        vocabs_not_w2v[word] = 1

                # If you have never seen a word, append it to vocab dictionary.
                if word not in vocabs:
                    vocabs_cnt += 1
                    vocabs[word] = vocabs_cnt
                    q2n.append(vocabs_cnt)
                else:
                    q2n.append(vocabs[word])

            # Append phrase as number representation
            df_raw.at[index, phrase] = q2n

    return df_raw, vocabs


def load_index_vector_dataframe(filename):
    dataframe = pd.read_csv(filename)

    for q in ['phrase1', 'phrase2']:
        dataframe[q + '_n'] = dataframe[q].apply(lambda x: [int(i) for i in x.replace('[', '').replace(']', '').split(',')])

    return dataframe


def load_embedding_matrix(filename):
    return np.load(filename)


def create_embedding_matrix(word_embedding, vocabs, embedding_dim=300):
    # This will be the embedding matrix
    embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)

    # The padding will be ignored
    embeddings[0] = 0

    # Build the embedding matrix
    for word, index in vocabs.items():
        if word in word_embedding.vocab:
            embeddings[index] = word_embedding.word_vec(word)

    # del word_embedding

    return embeddings


def load_word_embedding(word_embedding_type, empty_w2v=False):
    word_embedding_file = get_word_embedding_path_filename(word_embedding_type)

    print("Loading {} word embedding model (It may take a while)...".format(word_embedding_file))
    word2vec = __load_word_embedding(word_embedding_file, empty_w2v)
    print("word embedding loaded")

    return word2vec


def delete_word_embedding(word_embedding):
    del word_embedding


def save_embedding_matrix(embedding_matrix, dataset_type, word_embedding_type):
    filename = get_embedding_matrix_filename(dataset_type, word_embedding_type)
    np.save(filename, embedding_matrix)


def save_index_vector(df_index_vector, dataset_type, word_embedding_type, training_process=True):
    filename = get_index_vector_filename(dataset_type, word_embedding_type, training_process)
    df_index_vector.to_csv(filename, index=False, header=True)


def split_and_zero_padding(dataframe, max_seq_length):
    # Split to dicts
    side_phrases = {'left': dataframe['phrase1_n'], 'right': dataframe['phrase2_n']}
    dataset = None

    # Zero padding
    for dataset, side in itertools.product([side_phrases], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)

    return dataset


def find_max_seq_length(dataframe):
    max_seq_length = 0

    for column in dataframe[['phrase1_n', 'phrase2_n']]:
        series_phrase = dataframe[column].str.split().str.len()
        max_value = series_phrase.max()
        max_seq_length = (max_value if max_value > max_seq_length else max_seq_length)

    return int(max_seq_length)


def __get_dic_abbreviation_dataset_type():
    return {
        DatasetType.RAW: "raw",
        DatasetType.WITHOUT_SW: "sw",
        DatasetType.WITHOUT_SW_WITH_LEMMA: "sw-lemmatization"
    }


def __get_dic_abbreviation_word_embedding_type():
    return {
        WordEmbeddingType.WORD2VEC_WIKIPEDIA: "w2v_WIKI",
        WordEmbeddingType.WORD2VEC_GOOGLE_NEWS: "w2v_GN",
        WordEmbeddingType.GLOVE_WIKIPEDIA_GIGAWORD: "glove_WIKI_GIGA",
        WordEmbeddingType.GLOVE_COMMON_CRAWL_UNCASED: "glove_CC"
    }


def __text_to_word_list(text):
    # Pre process and convert texts to a list of words
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.split()


def __load_word_embedding(word_embedding_filename, empty):
    word_embedding = {}

    if empty:
        word_embedding = EmptyWord2Vec
        return word_embedding

    binary_file = True if word_embedding_filename.endswith(('.bin', '.bin.gz')) else False

    # Binary files are always word2vec pre-trained models otherwise they're glove
    if binary_file:
        word_embedding = KeyedVectors.load_word2vec_format(word_embedding_filename, binary=True)
    else:
        word_embedding = __load_word2vec_from_glove(word_embedding_filename)

    return word_embedding


def __load_word2vec_from_glove(glove_file):
    word2vec_tmp_file = os.path.join(DATA_FILES_WORD_EMBEDDINGS_PATH, "word2vec_tmp_file.txt")

    _ = glove2word2vec(glove_file, word2vec_tmp_file)
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_tmp_file, binary=False)

    if os.path.exists(word2vec_tmp_file):
        os.remove(word2vec_tmp_file)

    return word2vec_model


class EmptyWord2Vec:
    vocab = {}
    word_vec = {}
