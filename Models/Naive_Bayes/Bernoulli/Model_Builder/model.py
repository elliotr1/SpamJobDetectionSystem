from random import seed

from sklearn.model_selection import train_test_split

from Classification.Classifier import confusion_matrix_comparison, f1_score
from Models.Naive_Bayes.Bernoulli.Testing.testing import naiveBayesTestingFunc
from Models.Naive_Bayes.Bernoulli.Training.training_model import build_naive_bayes
import pandas as pd
def naive_bayes_func(df, bias=1, testing_data_size=.25):
    df, selected_columns = select_columns(df)
    selected_columns.remove("fraudulent")
    training_set, testing_set = split_data(df, testing_data_size)
    y_true = testing_set["fraudulent"]
    testing_set = testing_set[selected_columns]
    print("Training dataset")
    probabilities = build_naive_bayes(training_set, selected_columns)
    testing_set = testing_set[selected_columns]
    print("Running the testing set.")
    predictions = naiveBayesTestingFunc(probabilities, testing_set, selected_columns, bias=bias)
    evaluate_predictions(y_true, predictions, testing_set)


def evaluate_predictions(y_true, predictions, testing_set):
    return confusion_matrix_comparison(y_true, predictions, testing_set)


def select_columns(df):
    selected_columns = df.columns.tolist()
    df = df[selected_columns].dropna()
    return df, selected_columns


def split_data(df, test_size):
    training_set, testing_set = train_test_split(df, test_size=test_size, shuffle=True)
    return training_set, testing_set


def naive_bayes_mixed_data(df, unseen_datasets, bias=0.97, testing_data_size=.20):
    df, selected_columns = select_columns(df)
    selected_columns = df.columns.tolist()

    df = df[selected_columns].dropna()
    selected_columns.remove("fraudulent")
    training_set, testing_set = split_data(df, testing_data_size)
    print("Training dataset")
    probabilities = build_naive_bayes(training_set, selected_columns)
    y_true = testing_set["fraudulent"]
    testing_set = testing_set[selected_columns]

    print("Running seen testing set.")
    predictions = naiveBayesTestingFunc(probabilities, testing_set, selected_columns, bias=bias)
    without_unseen_data = f1_score(y_true, predictions)
    print(without_unseen_data)
    tables_used = ''
    for unseen_dataset in unseen_datasets:
        tables_used += unseen_dataset["table_name"] + ' '
        unseen_predictions = naiveBayesTestingFunc(probabilities, unseen_dataset["dataset"], selected_columns, bias=bias)
        print(f"Unseen predictions:\nHam: {predictions.count(0)}\nSpam: {predictions.count(1)}\n")
        unseen_dataset["dataset"]["fraudulent"] = unseen_predictions
        unseen_dataset["dataset"].to_csv("unseen_predictions.csv")
        df_combined = pd.concat([training_set, unseen_dataset["dataset"]], ignore_index=True)
        probabilities = build_naive_bayes(df_combined, selected_columns)

        testing_set = testing_set[selected_columns]
        print("Running unseen testing set.\n")
        predictions = naiveBayesTestingFunc(probabilities, testing_set, selected_columns, bias=bias)
        confusion_matrix_comparison(y_true, predictions, testing_set)
        loss = f1_score(y_true, predictions)

        print(f"Unseen datasets indexes used: {tables_used}")
        print(loss * 100, "% f1")
        print(round(100 - ((loss / without_unseen_data) * 100), 2), "% loss\n")
