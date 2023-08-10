from Models.Naive_Bayes.Bernoulli.Model_Builder.model import select_columns, split_data
from TextFormation.TextToBagOfWords import convert_all_columns_to_bow


def naive_bayes(dataset, testing_data_size=.25):
    df, selected_columns = select_columns(dataset)
    selected_columns = df.columns.tolist()
    dataset = df[selected_columns].dropna()
    training_set, testing_set = split_data(dataset, testing_data_size)
    ham = training_set[training_set["fraudulent"] == 0]
    spam = training_set[training_set["fraudulent"] == 1]
    ham_words = convert_all_columns_to_bow(ham)
    spam_words = convert_all_columns_to_bow(spam)
    ham_total_words_count = sum([sum(dic.values()) for dic in ham_words])
    spam_total_words_count = sum([sum(dic.values()) for dic in spam_words])
    ham_class_probability = ham_total_words_count / (ham_total_words_count + spam_total_words_count)
    spam_class_probability = spam_total_words_count / (ham_total_words_count + spam_total_words_count)


    print(1-(len(ham)/(len(ham)+len(spam))))
    print(1-(len(spam)/(len(ham)+len(spam))))

    print(1-ham_class_probability)
    print(1-spam_class_probability)
