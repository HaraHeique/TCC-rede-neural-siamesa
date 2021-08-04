# -*- coding: utf-8 -*-

"""
    It is responsible to handle user output information about training and prediction of the Neural Network.
"""

import os


def show_training_finished_message(n_epochs, training_start_time, training_end_time):
    time_in_ms = (training_end_time - training_start_time).microseconds / 1000
    print("Training time finished.\n%d epochs in %12.2f sec" % (n_epochs, time_in_ms))


def show_leaving_message():
    print("\nThanks for using. Have a nice day :)\n")


def clear_screen():
    try:
        os.system('cls' if os.name == 'nt' else 'clear')
    except Exception:
        pass


def break_lines(number_of_lines):
    for i in range(number_of_lines):
        print()


if __name__ == '__main__':
    break_lines(2)
    show_leaving_message()
