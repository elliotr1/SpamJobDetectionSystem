from TextFormation.TextToBagOfWords import convert_column_to_bow


def build_naive_bayes(training_set, columns):
    probabilities = {}
    total_ham = len(training_set[training_set["fraudulent"] == 0])
    total_spam = len(training_set[training_set["fraudulent"] == 1])

    for column in columns:

        probabilities[column] = {"spam": {}, "ham": {}}

        ham_counts = convert_column_to_bow(training_set[training_set["fraudulent"] == 0][column])
        spam_counts = convert_column_to_bow(training_set[training_set["fraudulent"] == 1][column])

        for word in spam_counts[1].keys():
            word_spam_probability = ((spam_counts[1][word]) + 1) / (
                    total_spam + spam_counts[0] + ham_counts[0])
            probabilities[column]["spam"][word] = word_spam_probability

        for word in ham_counts[1].keys():
            word_ham_probability = ham_counts[1][word] / (total_ham + spam_counts[0] + ham_counts[0])
            probabilities[column]["ham"][word] = word_ham_probability

    return probabilities
