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
import itertools
import pandas as pd
import nltk.tokenize as tokenize
import nltk.stem as stem
import nltk.corpus as corpus
import src.core.helper as helper


def extract_works_sentence_data(dic_works, n_sentences_per_author, qnt_sentences_by_works_filename):
    dic_data = {}
    qnt_works_by_author = {}
    __download_text_processing_depedencies()
    sent_tokenizer = __get_sent_tokenizer()

    for author, paths_works in dic_works.items():
        paths_works.sort()
        n_sentences_per_work = math.ceil(n_sentences_per_author / len(paths_works))
        works_greater_n_sentences_per_work = []

        for work in paths_works:
            raw_data = __read_data(work)
            sent_tokens = sent_tokenizer.tokenize(raw_data)
            filtered_tokens = __filter_sentence_tokens(sent_tokens)
            sentences_by_work = filtered_tokens[0:n_sentences_per_work]

            if author in dic_data:
                dic_data[author] += sentences_by_work
            else:
                dic_data[author] = sentences_by_work
                qnt_works_by_author[author] = {}

            qnt_works_by_author[author][work] = len(sentences_by_work)

            if len(filtered_tokens) > n_sentences_per_work:
                works_greater_n_sentences_per_work.append(work)

        if len(dic_data[author]) > n_sentences_per_author:
            # To set the amount of total sentences
            qnt_works_by_author[author][paths_works[-1]] -= (len(dic_works[author]) - n_sentences_per_author)
            dic_works[author] = dic_works[author][0:n_sentences_per_author]
        elif len(dic_data[author]) < n_sentences_per_author:
            # To complete the remaining sentences
            for work in works_greater_n_sentences_per_work:
                raw_data = __read_data(work)
                sent_tokens = sent_tokenizer.tokenize(raw_data)
                filtered_tokens = __filter_sentence_tokens(sent_tokens)
                sentences_by_work = filtered_tokens[n_sentences_per_work:]
                qnt_works_by_author[author][work] += len(sentences_by_work)
                dic_data[author] += sentences_by_work

                if len(dic_data[author]) > n_sentences_per_author:
                    qnt_works_by_author[author][work] -= (len(dic_data[author]) - n_sentences_per_author)
                    dic_data[author] = dic_data[author][0:n_sentences_per_author]
                    break

    # __export_length_sentences(sum(dic_data.values(), []), 100)
    __export_qnt_sentences_by_works(qnt_works_by_author, qnt_sentences_by_works_filename)

    return dic_data


def save_training_sentences_as_csv(dic_data_works, n_sentences_per_author, n_partitions):
    if not bool(dic_data_works):
        return

    # csv_filename = "training-{}-sentences.csv".format(n_sentences_per_author * n_authors * 2)
    csv_filename = "training-sentences.csv"
    id_count = 1
    columns = ['qd1', 'qd2', 'phrase1', 'phrase2', 'label']
    dic_dataframe = {'qd1': [], 'qd2': [], 'phrase1': [], 'phrase2': [], 'label': []}
    n_sentences_per_partition = math.floor(n_sentences_per_author / n_partitions)

    # Phrases of the same author (similarity = 1)
    for author, sentences in dic_data_works.items():
        length_sentences = n_sentences_per_partition * 2
        random.shuffle(sentences)

        for i in range(0, length_sentences, 2):
            dic_dataframe['qd1'].append(id_count)
            dic_dataframe['qd2'].append(id_count + 1)
            dic_dataframe['phrase1'].append(sentences.pop(0))
            dic_dataframe['phrase2'].append(sentences.pop(0))
            dic_dataframe['label'].append(1)
            id_count += 2

    # Phrases of the different author (similarity = 0)
    author_combinations = __get_author_combinations(list(dic_data_works.keys()))

    for combination in author_combinations:
        author_a = combination[0]
        author_b = combination[1]
        sentences_author_a = dic_data_works[author_a]
        sentences_author_b = dic_data_works[author_b]
        random.shuffle(sentences_author_a)
        random.shuffle(sentences_author_b)

        for i in range(0, n_sentences_per_partition, 1):
            dic_dataframe['qd1'].append(id_count)
            dic_dataframe['qd2'].append(id_count + 1)
            dic_dataframe['phrase1'].append(sentences_author_a.pop(0))
            dic_dataframe['phrase2'].append(sentences_author_b.pop(0))
            dic_dataframe['label'].append(0)
            id_count += 2

    # Export to csv file
    dataframe = pd.DataFrame(dic_dataframe, columns=columns)
    dataframe.to_csv(os.path.join(helper.DATA_FILES_TRAINING_PATH, csv_filename), index=False, header=True)


def save_prediction_sentences_as_csv(dic_data_works, n_sentences_per_author, n_partitions):
    if not bool(dic_data_works):
        return

    # csv_filename = "prediction-{}-sentences.csv".format(n_sentences_prediction)
    csv_filename = "prediction-sentences.csv"
    columns = ['phrase1', 'phrase2']
    dic_dataframe = {'phrase1': [], 'phrase2': []}
    n_sentences_per_partition = math.floor(n_sentences_per_author / n_partitions)

    # Phrases of the same authors
    for author, sentences in dic_data_works.items():
        length_sentences = n_sentences_per_partition * 2
        random.shuffle(sentences)

        for i in range(0, length_sentences, 2):
            dic_dataframe['phrase1'].append(sentences.pop(0))
            dic_dataframe['phrase2'].append(sentences.pop(0))

    # Phrases of different authors
    author_combinations = __get_author_combinations(list(dic_data_works.keys()))

    for combination in author_combinations:
        author_a = combination[0]
        author_b = combination[1]
        sentences_author_a = dic_data_works[author_a]
        sentences_author_b = dic_data_works[author_b]
        random.shuffle(sentences_author_a)
        random.shuffle(sentences_author_b)

        for i in range(0, n_sentences_per_partition):
            dic_dataframe['phrase1'].append(sentences_author_a.pop(0))
            dic_dataframe['phrase2'].append(sentences_author_b.pop(0))

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
        with open(filename, encoding="utf8", mode='r') as f:
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
    punkt_params = tokenize.punkt.PunktParameters()
    punkt_params.abbrev_types = set(__get_list_common_abrrev())
    tokenizer = tokenize.punkt.PunktSentenceTokenizer(punkt_params)

    return tokenizer


def __get_wordnet_pos(word):
    wordnet = corpus.wordnet

    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }

    return tag_dict.get(tag, wordnet.NOUN)


def __download_text_processing_depedencies():
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('stopwords')


def __filter_sentence_tokens(sent_tokens):
    sent_tokens = __slice_sentence_tokens(sent_tokens, 5)
    filtered_tokens = []
    # stemmer = stem.PorterStemmer() # Stemming
    lemmatizer = stem.WordNetLemmatizer()  # Lemmatization
    stopwords = set(corpus.stopwords.words('english'))

    for sentence in sent_tokens:
        if not __valid_sentence_token(sentence):
            continue

        # Split into words
        tokens = tokenize.word_tokenize(sentence)

        # Convert to lower case
        tokens = [w.lower() for w in tokens]

        # Stemming of words
        # tokens = [stemmer.stem(w) for w in tokens]

        # Lemmatization of words
        tokens_lemmatized = [lemmatizer.lemmatize(w, __get_wordnet_pos(w)) for w in tokens]

        # Remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens_lemmatized]

        # Remove remaining tokens that are not alphabetic
        word_list = [word for word in stripped if word.isalpha()]

        # Removing stopwords
        word_list = [word for word in word_list if word not in stopwords]

        # Join the words normalized with one common space
        normalized_sentence = ' '.join(word_list)

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


def __get_author_combinations(authors):
    combinations_object = itertools.combinations(authors, 2)
    combinations = list(combinations_object)

    return combinations


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


def __export_qnt_sentences_by_works(qnt_works_by_author, csv_filename):
    csv_data = []

    for author, works_obj in qnt_works_by_author.items():
        csv_data += list(map(lambda work: [author, work, works_obj[work]], works_obj))

    dataframe = pd.DataFrame(csv_data, columns=['author', 'work', 'sentences quantity'])
    dataframe.to_csv(os.path.join(helper.DATA_FILES_RESULTS_PATH, csv_filename), index=False)
