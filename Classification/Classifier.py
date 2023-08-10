import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import csv

from TextFormation.TextToWordList import convert_paragraph_to_list


def errorsAddToCSV(errors):
    error_words = []
    column_names = list(errors.columns)

    for _, error_row in errors.iterrows():
        error_dict = {}
        for column, value in error_row.items():
            error_dict[column] = convert_paragraph_to_list(value)
        error_words.append(error_dict)

    with open('error_words.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for index, error_dict in enumerate(error_words):
            if index > 0:
                writer.writerow([])  # Add an empty row before each new iteration
            for column in column_names:
                writer.writerow([f'{column} (index {index})'])
                writer.writerow(["Values"] + error_dict[column])


def confusion_matrix_comparison(y_true, y_pred, testing_set=None):
    indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred]
    errors = testing_set.iloc[indices]
    errorsAddToCSV(errors)

    confusion_matrix_model, accuracy, precision, recall, f1 = confusion_matrix_and_accuracies(y_true, y_pred)
    print("Accuracy: ", round(accuracy * 100, 2), "%")
    print("Precision: ", round(precision * 100, 2), "%")
    print("Recall: ", round(recall * 100, 2), "%")
    print("F1-Score: ", round(f1 * 100, 2), "%")
    display_cm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_model, display_labels=["Ham", "Spam"])
    display_cm.plot()
    # plt.show()


def confusion_matrix_and_accuracies(y_true, y_pred):
    confusion_matrix_model = confusion_matrix(y_true, y_pred)
    accuracy = (confusion_matrix_model[0, 0] + confusion_matrix_model[1, 1]) / len(y_true)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return [confusion_matrix_model, accuracy, precision, recall, f1_score]


def f1_score(y_true, y_pred):
    _, _, _, _, f1 = confusion_matrix_and_accuracies(y_true, y_pred)
    return f1
