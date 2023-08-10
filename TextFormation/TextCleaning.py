import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def lemmatize_word_in_list(word_list):
    return [lemmatizer.lemmatize(token) for token in word_list]


def remove_special_characters(word_list):
    structural_chars = ['.', ',', ';', ':', '!', '?']  # Add or modify the list of structural characters as needed
    cleaned_words = []
    for word in word_list:
        if len(word) >= 2:
            if word.isupper() or (word[0].isupper() and word[1:].islower()):
                cleaned_words.append(word)
            else:
                matches = re.findall(r'([a-zA-Z]+)([A-Z][a-z]*)', word)
                if matches:
                    first_word, second_word = matches[0]
                    cleaned_words.extend([first_word, second_word])
                else:
                    cleaned_words.append(word)

    return [re.sub(r'[^\w\s]', ' ', word).lower() if any(char in word for char in structural_chars) else re.sub(r'[^\w]', '', word).lower() for word in cleaned_words]



def remove_stopwords_from_list(word_list):
    return [token for token in word_list if token not in stop_words]
