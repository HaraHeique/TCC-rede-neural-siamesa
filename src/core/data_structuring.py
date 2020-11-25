# -*- coding: utf-8 -*-

"""
    Contains a set of reusable functions to read, pre-process and prepare data.
"""

import os
import sys
import string
import nltk
import math
import random
import pandas as pd
import nltk.tokenize as tokenize
import nltk.stem as porter
import src.core.helper as helper


def extract_works_sentence_data(dic_works, n_sentences_per_author):
    dic_data = {}
    sent_tokenizer = __get_sent_tokenizer()

    for author, paths_works in dic_works.items():
        for path in paths_works:
            raw_data = __read_data(path)
            sent_tokens = sent_tokenizer.tokenize(raw_data)
            filtered_tokens = __filter_sentence_tokens(sent_tokens)
            dic_data[author] = dic_data[author] + filtered_tokens if author in dic_data else [] + filtered_tokens

            if len(dic_data[author]) >= n_sentences_per_author:
                dic_data[author] = dic_data[author][0:n_sentences_per_author]
                break

    __export_length_sentences(sum(dic_data.values(), []), 100)

    return dic_data


def save_training_sentences_as_csv(dic_data_works, n_sentences_per_author):
    n_authors = len(dic_data_works.keys())
    csv_filename = "training-{}-sentences.csv".format(n_sentences_per_author * n_authors)
    id_count = 1
    columns = ['qd1', 'qd2', 'phrase1', 'phrase2', 'label']
    dic_dataframe = {'qd1': [], 'qd2': [], 'phrase1': [], 'phrase2': [], 'label': []}

    for author, sentences in dic_data_works.items():
        length_sentences = len(sentences)

        for i in range(0, length_sentences, 2):
            dic_dataframe['qd1'].append(id_count)
            dic_dataframe['qd2'].append(id_count + 1)
            dic_dataframe['phrase1'].append(sentences[i])
            dic_dataframe['phrase2'].append(sentences[i + 1])
            dic_dataframe['label'].append(author)
            id_count += 2

    dataframe = pd.DataFrame(dic_dataframe, columns=columns)
    dataframe.to_csv(os.path.join(helper.DATA_FILES_TRAINING_PATH, csv_filename), index=False, header=True)


def save_prediction_sentences_as_csv(dic_data_works, n_sentences):
    n_sentences_prediction = math.ceil(n_sentences)
    csv_filename = "prediction-{}-sentences.csv".format(n_sentences_prediction)
    columns = ['phrase1', 'phrase2']
    dic_dataframe = {'phrase1': [], 'phrase2': []}

    if not bool(dic_data_works):
        return

    # First column - Take n_sentences_prediction randomly from the first author
    first_author = next(iter(dic_data_works))
    random.shuffle(dic_data_works[first_author])
    dic_dataframe['phrase1'] = dic_data_works[first_author][0:n_sentences_prediction]

    # Second column - Take n_sentences_prediction randomly from authors inside dic_data_works
    random_works = []
    length_authors = len(dic_data_works.keys())
    n_sentences_per_author = math.ceil(n_sentences_prediction / length_authors)

    for author in dic_data_works.keys():
        sentences_per_author = [item for item in dic_data_works[author] if item not in dic_dataframe['phrase1']]
        random.shuffle(sentences_per_author)
        random_works += sentences_per_author[0:n_sentences_per_author]

    dic_dataframe['phrase2'] = random_works[0:n_sentences_prediction]

    # Export to csv file
    dataframe = pd.DataFrame(dic_dataframe, columns=columns)
    dataframe.to_csv(os.path.join(helper.DATA_FILES_PREDICTION_PATH, csv_filename), index=False, header=True)


def list_dir_authors(directory_name):
    dir_authors = []

    for dir in os.listdir(directory_name):
        if os.path.isdir(os.path.join(directory_name, dir)):
            dir_authors.append(os.path.join(directory_name, dir))

    return dir_authors


def works_by_author(path_author):
    works = []

    for file in os.listdir(path_author):
        if os.path.isfile(os.path.join(path_author, file)):
            works.append(os.path.join(path_author, file))

    return works


def dic_works_by_authors(path_authors):
    works = {}

    for path in path_authors:
        author = __get_author_by_path(path)
        works[author] = [] + works_by_author(path)

    return works


def __read_data(filename):
    try:
        with open(filename, mode='r') as f:
            raw_data = f.read()
    except FileNotFoundError as err:
        print(err)
        sys.exit(1)
    except IOError as err:
        print(err)
        sys.exit(1)
    except Exception as err:
        print(err)
        sys.exit(1)
    else:
        return raw_data


def __get_list_common_abrrev():
    return [
        "Mr", "Mrs", "LLC", "Pres", "approx", "min", "vs", "E.T.A", "dept", "c/o", "B.Y.O.B", "apt", "appt", "A.S.A.P",
        "D.I.Y", "est", "vet", "temp", "R.S.V.P"
    ]


def __get_sent_tokenizer():
    nltk.download('punkt')
    punkt_params = tokenize.punkt.PunktParameters()
    punkt_params.abbrev_types = set(__get_list_common_abrrev())
    tokenizer = tokenize.punkt.PunktSentenceTokenizer(punkt_params)

    return tokenizer


def __filter_sentence_tokens(sent_tokens):
    sent_tokens = __slice_sentence_tokens(sent_tokens, 5)
    filtered_tokens = []
    stemmer = porter.PorterStemmer()

    for sentence in sent_tokens:
        if not __valid_sentence_token(sentence):
            continue

        # Split into words
        tokens = tokenize.word_tokenize(sentence)

        # Convert to lower case
        tokens = [w.lower() for w in tokens]

        # Stemming of words
        stemmed_tokens = [stemmer.stem(w) for w in tokens]

        # Remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in stemmed_tokens]

        # Remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]

        # Join the words normalized with one common space
        normalized_sentence = ' '.join(words)

        # Check if the sentence normalized is valid
        if not __valid_sentence_token(normalized_sentence):
            continue

        # Add the sentence normalized to list
        filtered_tokens.append(normalized_sentence)

    return filtered_tokens


def __valid_sentence_token(sent_token):
    if '\n' in sent_token:
        return False

    if len(sent_token.strip()) <= 0 or len(sent_token.split()) > 150:
        return False

    return True


def __slice_sentence_tokens(sent_tokens, percent):
    length_sentences = len(sent_tokens)

    if sent_tokens is None or length_sentences <= 0:
        return []

    abs_percent = percent / 100
    abs_units = math.ceil(length_sentences * abs_percent)

    start_index = abs_units
    end_index = (length_sentences - abs_units) + 1
    result = sent_tokens[start_index:end_index]

    return result


def __get_author_by_path(path):
    return path.split('/')[-1]


def __export_length_sentences(tokens, length_sentences):
    dic_dataframe = {'length': [], 'sentence': []}
    sorted_tokens = sorted(tokens, key=lambda sentence: len(sentence.split()), reverse=True)

    for s in sorted_tokens:
        length_current_sentence = len(s.split())

        if length_current_sentence >= length_sentences:
            dic_dataframe['length'].append(length_current_sentence)
            dic_dataframe['sentence'].append(s)

    csv_filename = "sentences-greater-{}.csv".format(length_sentences)
    dataframe = pd.DataFrame(dic_dataframe, columns=['length', 'sentence'])
    dataframe.to_csv(os.path.join(helper.DATA_FILES_RESULTS_PATH, csv_filename), index=False, header=True)
