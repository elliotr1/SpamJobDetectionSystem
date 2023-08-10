import numpy as np

from Models.Naive_Bayes.Bernoulli.Testing.BernoulliCode.Bernoulli import bernoulli_naive_bayes_function
from Models.Naive_Bayes.Bernoulli.Testing.laplace.laplace import getLapLaceTuningForcolumn

def naiveBayesTestingFunc(training_probabilities, testing_set, columns, bias=1.3):
    overall_predictions = []
    for i in range(len(testing_set)):
        row = testing_set.iloc[i]
        column_predictions = {"spam": [], "ham": []}
        for column in columns:
            k, word_list = getLapLaceTuningForcolumn(row, column, training_probabilities)
            word_probabilities = {"spam": [], "ham": []}
            word_probabilities, column_predictions = bernoulli_naive_bayes_function(
                word_list=word_list, training_probabilities=training_probabilities,
                column_name=column, k=k, column_predictions=column_predictions,
                word_probabilities=word_probabilities)
        if abs(sum(column_predictions["ham"])) * bias > abs(sum(column_predictions["spam"])):
            overall_predictions.append(1)
        else:
            overall_predictions.append(0)
    return overall_predictions



def NaiveBayesTestingActiveLearning(training_probabilities, testing_set, columns, bias=1.3):
    spamVariance = []
    hamVariance = []
    overall_predictions = []
    for i in range(len(testing_set)):
        row = testing_set.iloc[i]
        column_predictions = {"spam": [], "ham": []}
        for column in columns:
            k, word_list = getLapLaceTuningForcolumn(row, column, training_probabilities)
            word_probabilities = {"spam": [], "ham": []}
            word_probabilities, column_predictions = bernoulli_naive_bayes_function(
                word_list=word_list, training_probabilities=training_probabilities,
                column_name=column, k=k, column_predictions=column_predictions,
                word_probabilities=word_probabilities)
        hamVariance.append({"Index": i, "variance": np.array(column_predictions["ham"]).var(), "rowData": testing_set[i]})
        spamVariance.append({"Index": i, "variance": np.array(column_predictions["spam"]).var(), "rowData": testing_set[i]})
        if abs(sum(column_predictions["ham"])) * bias > abs(sum(column_predictions["spam"])):
            overall_predictions.append(1)
        else:
            overall_predictions.append(0)
    hamVariance = sorted(hamVariance, key=lambda item: hamVariance["variance"])
    spamVariance = sorted(spamVariance, key=lambda item: spamVariance["variance"])
    top_bottom_10_percent_hamVariance = hamVariance[:int(math.floor(len(hamVariance)*.1))]
    top_bottom_10_percent_spamVariance = spamVariance[:int(math.floor(len(spamVariance)*.1))]
    return [overall_predictions, top_bottom_10_percent_hamVariance, top_bottom_10_percent_spamVariance]
