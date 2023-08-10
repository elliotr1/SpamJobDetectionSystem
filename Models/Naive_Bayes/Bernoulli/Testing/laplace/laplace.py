from TextFormation.TextToBagOfWords import convert_column_to_bow
from TextFormation.TextToWordList import convert_paragraph_to_list


def getLapLaceTuningForcolumn(row, column, training_probabilities):
    total_amount_of_words_in_column = convert_column_to_bow(column)[0]
    word_list = convert_paragraph_to_list(row[column])
    k = laplaceSmoothing(len(word_list), total_amount_of_words_in_column, training_probabilities, column)
    return k, word_list


def laplaceSmoothing(word_count, total_words, training_probabilities, column, k=1):
    num_words_in_ham = len(training_probabilities[column]["ham"].keys())
    return (word_count + k) / (total_words + k * num_words_in_ham)
