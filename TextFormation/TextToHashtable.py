from TextFormation import TextToWordList


def convert_text_to_hashtable(column):
    list_of_words = list(set(TextToWordList.convert_column_to_list(column)))
    return {word: i + 1 for i, word in enumerate(list_of_words)}


def convert_all_columns_to_hashtable(dataset):
    return [convert_text_to_hashtable(dataset[column]) for column in dataset.columns]
