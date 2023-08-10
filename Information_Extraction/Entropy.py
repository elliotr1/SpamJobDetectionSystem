from cmath import e, log
from TextFormation.TextToBagOfWords import convert_column_to_bow
import numpy as np


def calculate_the_columns_entropy(column, base=2):
    total_amount_of_words, bow = convert_column_to_bow(column)
    number_of_instances_list_array = np.array([word_instance for word_instance in bow.values()])
    word_probabilities = number_of_instances_list_array / total_amount_of_words
    none_zeros = np.count_nonzero(word_probabilities)
    if none_zeros <= 1:
        return 0
    base = e if base is None else base
    bag_of_words = {word: -(i * log(i, base).real) for i, word in zip(word_probabilities, bow.keys())}
    parent_entropy = sum(bag_of_words.values())
    return [{k: v for k, v in sorted(bag_of_words.items(), key=lambda item: item[1])}, parent_entropy]
