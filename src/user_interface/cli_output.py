# -*- coding: utf-8 -*-

"""
    It is responsible to handle user output information about training and prediction of the Neural Network.
"""

import os


def training_finished_message(n_epoch, training_start_time, training_end_time):
    print("Training time finished.\n%d epochs in %12.2f" % (n_epoch, training_end_time - training_start_time))


def clear_screen():
    try:
        os.system('cls' if os.name == 'nt' else 'clear')
    except Exception:
        pass
