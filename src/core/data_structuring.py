# -*- coding: utf-8 -*-

"""
    Contains a set of reusable functions to read, pre-process and prepare data.
"""

import pandas as pd
import nltk


def read_data(filename):
    with open(filename, mode='r') as f:
        raw_data = f.read()
        nltk.downloader('punkt')
