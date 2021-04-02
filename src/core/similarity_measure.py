# -*- coding: utf-8 -*-

"""
    Contains a set of functions to performs the similarity measure of the output Neural Network.
"""

from tensorflow.python.keras import backend as k
from tensorflow.python.keras.layers import Lambda, Reshape, dot


def calculate_manhattan_distance(left_output, right_output):
    manhattan_distance = Lambda(
        function=lambda x: __exponent_neg_manhattan_distance(x[0], x[1]),
        output_shape=lambda x: (x[0][0], 1)
    )([left_output, right_output])

    return manhattan_distance


def calculate_cosine_distance(left_output, right_output):
    cos_distance = dot([left_output, right_output], axes=1, normalize=True)
    cos_distance = Reshape((1,))(cos_distance)
    cos_similarity = Lambda(lambda x: 1 - x)(cos_distance)

    return cos_similarity


def calculate_cosine_distance_from_vectors(vects):
    x, y = vects
    x = k.l2_normalize(x, axis=-1)
    y = k.l2_normalize(y, axis=-1)
    return -k.mean(x * y, axis=-1, keepdims=True)


def calculate_euclidean_distance(vects):
    x, y = vects
    sum_square = k.sum(k.square(x - y), axis=1, keepdims=True)

    return k.sqrt(k.maximum(sum_square, k.epsilon()))


def calculate_jaccard_distance(vects):
    x, y = vects
    smooth = 100

    intersection = k.sum(k.abs(x * y), axis=-1)
    sum_ = k.sum(k.abs(x) + k.abs(y), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return (1 - jac) * smooth


def dist_output_shape(shapes):
    shape1, shape2 = shapes

    return (shape1[0], 1)


def __exponent_neg_manhattan_distance(left, right):
    return k.exp(-k.sum(k.abs(left - right), axis=1, keepdims=True))

