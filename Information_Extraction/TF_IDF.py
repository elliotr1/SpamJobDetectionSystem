import math

import numpy
from TextFormation.TextToBagOfWords import convert_column_to_bow

import math


def tf_idf(column):
    bow = convert_column_to_bow(column)
    total_documents = bow[0]
    word_frequencies = bow[1]

    for word in word_frequencies:
        tf = word_frequencies[word] / total_documents

        idf = math.log(total_documents / (1 + word_frequencies[word]))

        tf_idf_score = tf * idf

        word_frequencies[word] = tf_idf_score

    print(word_frequencies)

