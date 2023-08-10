import time
import numpy as np
import tensorflow
from gensim.models.word2vec import Word2Vec
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dropout
from keras.src.optimizers import Adam
from scipy import sparse
from sklearn.model_selection import train_test_split

from Classification.Classifier import confusion_matrix_comparison
from Models.Naive_Bayes.Bernoulli.Model_Builder.model import select_columns
from TextFormation.TextToWordList import convert_paragraph_to_list


def normalize_and_pad_sequences(sequences):
    max_length = max(len(seq) for seq in sequences)

    # Initialize an array to hold normalized and padded sequences
    result_array = np.zeros((len(sequences), max_length, sequences[0][0].shape[0]))

    for i, seq in enumerate(sequences):
        seq_length = len(seq)
        # Normalize the sequence and copy it to the result array
        result_array[i, :seq_length, :] = seq / (np.linalg.norm(seq, axis=1, keepdims=True) + 1e-8)
        # Pad the sequence with zeros if needed
        if seq_length < max_length:
            result_array[i, seq_length:, :] = np.zeros((max_length - seq_length, sequences[0][0].shape[0]))

    return result_array


def recursiveNN(df):
    w2v = Word2Vec.load("word2vec.model")
    df, selected_columns = select_columns(df)
    df = df.sample(5000)
    selected_columns.remove("fraudulent")
    ham_data = df[df["fraudulent"] == 0]
    spam_data = df[df["fraudulent"] == 1]

    ham_vectors = []
    spam_vectors = []
    print("Converting words into w2v values.")
    start_time = time.time()
    total_columns = len(selected_columns)
    for i, column in enumerate(selected_columns):
        ham_data_column = [convert_paragraph_to_list(row) for row in ham_data[column]]
        spam_data_column = [convert_paragraph_to_list(row) for row in spam_data[column]]

        ham_column_vectors = [
            [w2v.wv[word] if word in w2v.wv else np.zeros(w2v.vector_size) for word in row]
            for row in ham_data_column
        ]

        spam_column_vectors = [
            [w2v.wv[word] if word in w2v.wv else np.zeros(w2v.vector_size) for word in row]
            for row in spam_data_column
        ]

        max_length = max(max(len(seq) for seq in ham_column_vectors), max(len(seq) for seq in spam_column_vectors))

        padded_ham_vectors = [
            seq + [np.zeros(w2v.vector_size)] * (max_length - len(seq))
            for seq in ham_column_vectors
        ]
        padded_spam_vectors = [
            seq + [np.zeros(w2v.vector_size)] * (max_length - len(seq))
            for seq in spam_column_vectors
        ]

        ham_vectors.extend(normalize_and_pad_sequences(padded_ham_vectors))
        spam_vectors.extend(normalize_and_pad_sequences(padded_spam_vectors))

        # Track progress
        progress = (i + 1) / total_columns * 100
        elapsed_time = time.time() - start_time
        remaining_time = elapsed_time / (progress / 100) - elapsed_time
        print(
            f"Processed column {i + 1}/{total_columns} | Progress: {progress:.2f}% | Elapsed Time: {elapsed_time:.2f}s | Remaining Time: {remaining_time:.2f}s")

    ham_column_sparse_array = sparse.vstack(map(sparse.csr_matrix, ham_vectors))
    spam_column_sparse_array = sparse.vstack(map(sparse.csr_matrix, spam_vectors))

    ham_labels = np.zeros(ham_column_sparse_array.shape[0])
    spam_labels = np.ones(spam_column_sparse_array.shape[0])

    training_data = sparse.vstack((ham_column_sparse_array, spam_column_sparse_array))
    labels = np.concatenate((ham_labels, spam_labels), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(training_data, labels, test_size=0.25, random_state=42)
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    """    reshaped_data = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    pca = PCA(n_components=20)
    reduced_data = pca.fit_transform(reshaped_data)
    print(reduced_data)"""

    model = Sequential()
    model.add(LSTM(units=128, input_shape=(X_train.shape[1], 1), return_sequences=True))
    model.add(LSTM(units=256, activation='sigmoid', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128, activation='sigmoid', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='tanh'))
    model.compile(optimizer=Adam(learning_rate=0.05), loss='binary_crossentropy')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    path = "savedModel.ckpt"
    cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True,
                                                             save_weights_only=False, verbose=1)

    model.fit(
        X_train, y_train, epochs=3, batch_size=96, validation_data=(X_test, y_test),
        callbacks=[early_stopping, cp_callback])
    predictions = model.predict(X_test)
    print(predictions)
    predictions = np.where(predictions >= 0.5, 1, 0)

    testing_predictions = predictions[:len(y_test)]

    confusion_matrix_comparison(y_test, testing_predictions)
