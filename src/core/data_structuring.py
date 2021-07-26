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
import statistics
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import nltk.tokenize as tokenize
import nltk.stem as stem
import nltk.corpus as corpus
import src.core.helper as helper
from src.enums.DatasetType import DatasetType


def extract_works_sentence_data(dic_works, n_sentences_per_author, dataset_type, isTraining):
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
            filtered_tokens = __filter_sentence_tokens(sent_tokens, dataset_type)
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
                filtered_tokens = __filter_sentence_tokens(sent_tokens, dataset_type)
                sentences_by_work = filtered_tokens[n_sentences_per_work:]
                qnt_works_by_author[author][work] += len(sentences_by_work)
                dic_data[author] += sentences_by_work

                if len(dic_data[author]) > n_sentences_per_author:
                    qnt_works_by_author[author][work] -= (len(dic_data[author]) - n_sentences_per_author)
                    dic_data[author] = dic_data[author][0:n_sentences_per_author]
                    break

    # __export_length_sentences(sum(dic_data.values(), []), 100)
    __export_qnt_sentences_by_works(qnt_works_by_author, dataset_type, isTraining)

    return dic_data


def save_training_sentences_as_csv(dic_data_works, dataset_type, n_sentences_per_author):
    if not bool(dic_data_works):
        return

    csv_filename = helper.get_dataset_type_filename(dataset_type, "training-sentences-{dataset_type}.csv")
    id_count = 1
    columns = ['qd1', 'qd2', 'phrase1', 'phrase2', 'label', 'author1', 'author2']
    dic_dataframe = {'qd1': [], 'qd2': [], 'phrase1': [], 'phrase2': [], 'label': [], 'author1': [], 'author2': []}

    # Phrases of the same author (similarity = 1)
    n_sentences_by_same_author = math.floor(n_sentences_per_author / 2)

    for author, sentences in dic_data_works.items():
        # random.shuffle(sentences)

        for i in range(0, n_sentences_by_same_author, 2):
            dic_dataframe['qd1'].append(id_count)
            dic_dataframe['qd2'].append(id_count + 1)
            dic_dataframe['phrase1'].append(sentences.pop(0))
            dic_dataframe['phrase2'].append(sentences.pop(0))
            dic_dataframe['label'].append(1)
            dic_dataframe['author1'].append(author)
            dic_dataframe['author2'].append(author)
            id_count += 2

    # Phrases of the different author (similarity = 0)
    author_combinations, n_combination_per_author = __get_author_combinations(list(dic_data_works.keys()))
    n_sentences_by_combination = math.floor(
        (n_sentences_per_author - n_sentences_by_same_author) / n_combination_per_author)

    for combination in author_combinations:
        author_a = combination[0]
        author_b = combination[1]
        sentences_author_a = dic_data_works[author_a]
        sentences_author_b = dic_data_works[author_b]
        # random.shuffle(sentences_author_a)
        # random.shuffle(sentences_author_b)

        for i in range(0, n_sentences_by_combination, 1):
            dic_dataframe['qd1'].append(id_count)
            dic_dataframe['qd2'].append(id_count + 1)
            dic_dataframe['phrase1'].append(sentences_author_a.pop(0))
            dic_dataframe['phrase2'].append(sentences_author_b.pop(0))
            dic_dataframe['label'].append(0)
            dic_dataframe['author1'].append(author_a)
            dic_dataframe['author2'].append(author_b)
            id_count += 2

    # Export to csv file
    dataframe = pd.DataFrame(dic_dataframe, columns=columns)
    dataframe.to_csv(os.path.join(helper.DATA_FILES_TRAINING_PATH, csv_filename), index=False, header=True)

    return dataframe


def save_prediction_sentences_as_csv(dic_data_works, dataset_type, n_sentences_per_author):
    if not bool(dic_data_works):
        return

    csv_filename = helper.get_dataset_type_filename(dataset_type, "prediction-sentences-{dataset_type}.csv")
    columns = ['phrase1', 'phrase2', 'author1', 'author2']
    dic_dataframe = {'phrase1': [], 'phrase2': [], 'author1': [], 'author2': []}

    # Phrases of the same authors
    n_sentences_by_same_author = math.floor(n_sentences_per_author / 2)

    for author, sentences in dic_data_works.items():
        # random.shuffle(sentences)

        for i in range(0, n_sentences_by_same_author, 2):
            dic_dataframe['phrase1'].append(sentences.pop(0))
            dic_dataframe['phrase2'].append(sentences.pop(0))
            dic_dataframe['author1'].append(author)
            dic_dataframe['author2'].append(author)

    # Phrases of different authors
    author_combinations, n_combination_per_author = __get_author_combinations(list(dic_data_works.keys()))
    n_sentences_by_combination = math.floor(
        (n_sentences_per_author - n_sentences_by_same_author) / n_combination_per_author)

    for combination in author_combinations:
        author_a = combination[0]
        author_b = combination[1]
        sentences_author_a = dic_data_works[author_a]
        sentences_author_b = dic_data_works[author_b]
        # random.shuffle(sentences_author_a)
        # random.shuffle(sentences_author_b)

        for i in range(0, n_sentences_by_combination):
            dic_dataframe['phrase1'].append(sentences_author_a.pop(0))
            dic_dataframe['phrase2'].append(sentences_author_b.pop(0))
            dic_dataframe['author1'].append(author_a)
            dic_dataframe['author2'].append(author_b)

    # Export to csv file
    dataframe = pd.DataFrame(dic_dataframe, columns=columns)
    dataframe.to_csv(os.path.join(helper.DATA_FILES_PREDICTION_PATH, csv_filename), index=False, header=True)

    return dataframe


def plot_hist_length_dataframe(dataframe, dataset_type):
    distribution_save_filename = helper.get_dataset_type_filename(dataset_type,
                                                                  "sentences-distribution-{dataset_type}.png")
    data = __phrases_dataframe_size(dataframe)

    plt.figure(figsize=[10, 8])
    plt.hist(x=data, bins=10, color='#D24324', alpha=0.9, rwidth=1)
    plt.xlabel("Sentence Length", fontsize=15)
    # plt.xticks(np.arange(0, max(data) + 1, 20), fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel("Frequency", fontsize=15)
    plt.title("Sentences Distribution Histogram", fontsize=15)
    plt.legend(handles=__statistics_mpatches(data), handlelength=0, handletextpad=0, fancybox=True)
    plt.savefig(os.path.join(helper.get_results_path_directory_by_dataset(dataset_type), distribution_save_filename))


def validate_total_number_of_sentences(dic_data_works, n_sentences_per_author, n_authors, dataset_type, isTraining):
    process_name = "training" if isTraining else "prediction"
    total_sentences = n_sentences_per_author * n_authors
    total_extracted_sentences = len(sum(dic_data_works.values(), []))

    if total_extracted_sentences < total_sentences:
        raise Exception('''
            Dataset Type: {dataset_type}
            
            The number of {process_name} sentences from authors is not enough. 
            Please enter a number of sentences for each author that supports the amount of extracted sentences.
            
            - Total sentences: {total_sentences}
            - Total extracted sentences: {extracted_sentences}'''
                        .format(dataset_type=dataset_type.name, process_name=process_name,
                                total_sentences=total_sentences,
                                extracted_sentences=total_extracted_sentences)
                        )


def list_dir_authors(directory_name, number_authors):
    dir_authors = []

    for dir in os.listdir(directory_name):
        if len(dir_authors) == number_authors:
            break

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


def __filter_sentence_tokens(sent_tokens, dataset_type):
    sent_tokens = __slice_sentence_tokens(sent_tokens, 5)
    filtered_tokens = []
    # stemmer = stem.PorterStemmer() # Stemming
    lemmatizer = stem.WordNetLemmatizer()  # Lemmatization
    stopwords = set(corpus.stopwords.words('english'))

    for sentence in sent_tokens:
        if not __valid_sentence_token(sentence):
            continue

        # Split into words
        tokens = tokenize.word_tokenize(sentence.strip())

        # Convert to lower case
        tokens = [w.lower() for w in tokens]

        if __should_lemmatize(dataset_type):
            # Stemming of words
            # tokens = [stemmer.stem(w) for w in tokens]

            # Lemmatization of words
            tokens = [lemmatizer.lemmatize(w, __get_wordnet_pos(w)) for w in tokens]

        # Remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]

        # Remove remaining tokens that are not alphabetic
        word_list = [word for word in stripped if word.isalpha()]

        if __should_remove_stopwords(dataset_type):
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

    if len(sent_token.split()) <= 1 or len(sent_token.split()) > 150:
        return False

    return True


def __should_remove_stopwords(dataset_type):
    return dataset_type == DatasetType.WITHOUT_SW or \
           dataset_type == DatasetType.WITHOUT_SW_WITH_LEMMA


def __should_lemmatize(dataset_type):
    return dataset_type == DatasetType.WITHOUT_SW_WITH_LEMMA


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

    # checking the number of times a token of a combination appears
    n_combinations_per_token = 0

    if len(combinations) > 0:
        token_to_check = combinations[0][0]

        for combination in combinations:
            is_in_combination = combination[0] == token_to_check or combination[1] == token_to_check
            n_combinations_per_token += 1 if is_in_combination else 0

    return combinations, n_combinations_per_token


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


def __export_qnt_sentences_by_works(qnt_works_by_author, dataset_type, isTraining):
    csv_data = []

    csv_template_filename = "quantity-sentences-by-works-{}.csv"
    csv_filename = csv_template_filename.format("training") if isTraining else csv_template_filename.format(
        "prediction")

    for author, works_obj in qnt_works_by_author.items():
        csv_data += list(map(lambda work: [author, work, works_obj[work]], works_obj))

    dataframe = pd.DataFrame(csv_data, columns=['author', 'work', 'sentences quantity'])
    dataframe.to_csv(os.path.join(helper.get_results_path_directory_by_dataset(dataset_type), csv_filename),
                     index=False)


def __statistics_mpatches(data):
    qnt_patch = mpatches.Patch(label="Qnt. - {qnt}".format(qnt=len(data)))
    min_patch = mpatches.Patch(label="Min. - {min}".format(min=int(min(data))))
    max_patch = mpatches.Patch(label="Max. - {max}".format(max=int(max(data))))
    mean_patch = mpatches.Patch(label="Mean - {:.2f}".format(statistics.mean(data)))
    mode_patch = mpatches.Patch(label="Mode - {mode}".format(mode=statistics.mode(data)))
    median_patch = mpatches.Patch(label="Median - {:.2f}".format(statistics.median(data)))
    stddev_patch = mpatches.Patch(label="Std. Dev. - {:.2f}".format(statistics.stdev(data)))
    variance_patch = mpatches.Patch(label="Variance - {:.2f}".format(statistics.variance(data)))

    patches = [qnt_patch, min_patch, max_patch, mean_patch, mode_patch, median_patch, stddev_patch, variance_patch]

    return patches


def __phrases_dataframe_size(dataframe):
    phrases_size = []

    for column in dataframe[['phrase1_n', 'phrase2_n']]:
        series_phrase = dataframe[column].str.split().str.len().fillna(0)
        phrases_size.extend(series_phrase.tolist())

    # for i, row in dataframe.iterrows():
    #    sentence_length = len(row['phrase1_n'].split()) if row['phrase1_n'] else 0
    #    phrases_size.append(sentence_length)

    #    sentence_length = len(row['phrase2_n'].split()) if row['phrase2_n'] else 0
    #   phrases_size.append(sentence_length)

    return phrases_size
