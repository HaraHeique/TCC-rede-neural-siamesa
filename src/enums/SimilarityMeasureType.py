from enum import Enum


class SimilarityMeasureType(Enum):
    MANHATTAN = 1
    EUCLIDEAN = 2
    COSINE = 3
    JACCARD = 4
