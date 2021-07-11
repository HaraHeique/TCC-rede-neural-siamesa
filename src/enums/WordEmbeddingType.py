from enum import Enum


class WordEmbeddingType(Enum):
    WORD2VEC_WIKIPEDIA = 1
    WORD2VEC_GOOGLE_NEWS = 2
    GLOVE_WIKIPEDIA_GIGAWORD = 3
    GLOVE_COMMON_CRAWL_UNCASED = 4