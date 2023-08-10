import numpy as np


def bernoulli_naive_bayes_function(word_list, training_probabilities, column_name, k, column_predictions,
                                   word_probabilities):
    ham_probabilities = training_probabilities[column_name]["ham"]
    spam_probabilities = training_probabilities[column_name]["spam"]
    total_ham_prob = sum(ham_probabilities.values())
    total_spam_prob = sum(spam_probabilities.values())
    num_ham_words = len(ham_probabilities)
    num_spam_words = len(spam_probabilities)

    for word in word_list:
        if word in ham_probabilities:
            word_count = ham_probabilities[word] + k
            ham_probability = word_count / (total_ham_prob + k * num_ham_words)
            word_probabilities["ham"].append(np.log(ham_probability))
        else:
            column_predictions["ham"].append(np.log(k / (k * num_ham_words)))

        if word in spam_probabilities:
            word_count = spam_probabilities[word] + k
            spam_probability = word_count / (total_spam_prob + k * num_spam_words)
            word_probabilities["spam"].append(np.log(spam_probability))
        else:
            column_predictions["spam"].append(np.log(k / (k * num_spam_words)))

    return word_probabilities, column_predictions
