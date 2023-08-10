from TextFormation.TextCleaning import lemmatize_word_in_list, remove_stopwords_from_list, remove_special_characters


def convert_column_to_list(column):
    words = []

    for text_contents in column:
        if type(text_contents) == str:
            if " " in text_contents:
                tokens = text_contents.split()
            else:
                tokens = text_contents
            removed_special_characters_words = remove_special_characters(tokens)
            lemmatized_words = lemmatize_word_in_list(removed_special_characters_words)
            filtered_tokens = remove_stopwords_from_list(lemmatized_words)
            filtered_tokens = [token.replace(" ", "") for token in filtered_tokens if len(token)>1]
            words.extend(filtered_tokens)
    return words


def convert_paragraph_to_list(paragraph):

    tokens = paragraph.split()
    removed_special_characters_words = remove_special_characters(tokens)
    lemmatized_words = lemmatize_word_in_list(removed_special_characters_words)
    filtered_tokens = remove_stopwords_from_list(lemmatized_words)
    return filtered_tokens
