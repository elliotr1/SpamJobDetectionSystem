from collections import defaultdict, Counter

from TextFormation.TextCleaning import lemmatize_word_in_list, remove_stopwords_from_list, remove_special_characters
from TextFormation.TextToWordList import convert_column_to_list


def convert_column_to_bow(column):
    words = convert_column_to_list(column)
    words = [word for word in words if len(word) > 1]
    bow = defaultdict(int)
    bow.update(Counter(words))
    bow = {k: v for k, v in sorted(bow.items(), key=lambda item: item[1])}
    return [sum(bow.values()), bow]


def convert_all_columns_to_bow(dataset):
    return [convert_column_to_bow(dataset[column])[1] for column in dataset.columns]
