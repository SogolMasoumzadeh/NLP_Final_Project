from datetime import datetime
from typing import List

from tensorflow.keras import layers
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.models import Sequential, Model
import csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

JOKES_PATH = "jokes_processed_20201110.csv"
YAHOO_PATH = "df_yahoo_news_20201123.csv"
GLOVE_PATH = "glove/glove.6B.50d.txt"
MAXLEN = 100
DIMENSION = 50
TRAIN_EPOCH = 1
TRAIN_BATCHSIZE = 10
TEST_BATCHSIZE = 10


class classifer():

    def __init__(self):
        self.__keras_tokenizer = Tokenizer(num_words=5000)
        self.__corpus = []
        self.__corpus_labels = []
        self.__train_data = []
        self.__test_data = []
        self.__train_label = []
        self.__test_label = []
        self.__dev_data = []
        self.__dev_label = []
        self.__tokenized_train = []
        self.__tokenized_test = []
        self.__tokenized_dev = []

        self.__glove_embedding = None
        self.__jokes_path = None
        self.__non_jokes_path = None
        self.__glove_path = None
        self.__cnn_history = None
        self.__cnn_results = None
        self.__cnn_predictions = None

    def __file_path_creator(self, file_name: str):
        """Create the path to the jokes and non jokes file and the glove embeddings..."""
        return os.path.abspath(file_name)


    def __data_loader(self, file_path: str, joke_flag: bool):
        """Data loader from the .csv files"""
        df = []
        label = []

        if joke_flag:
            with open(file_path, encoding="utf-8") as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                for row in readCSV:
                    df.append(row[1])  # TODO: add more fields for jokes?
                    label.append(1)
        if not joke_flag:
            with open(file_path, encoding="utf-8") as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                for row in readCSV:
                    df.append(row[1])
                    label.append(0)
        return df, label

    def __corpus_creator(self, joke_list, non_joke_list, joke_labels: List[int], non_joke_labels: List[int]):
        non_joke_list.extend(joke_list)
        non_joke_labels.extend(joke_labels)
        self.__corpus = non_joke_list
        self.__corpus_labels = non_joke_labels

    def __train_test_divider(self, data_file: List[str], data_label):
        """Splitting the data between train and test"""
        self.__train_data, self.__test_data, self.__train_label, self.__test_label = \
            train_test_split(data_file, data_label, test_size=0.1, random_state=1000)

        self.__train_data, self.__dev_data, self.__train_label, self.__dev_label = \
            train_test_split(self.__train_data, self.__train_label, test_size=0.15 / (0.85 + 0.15), random_state=1000)

    def __keras_tokenize(self):
        """Tokenize the train and test datasets"""
        self.__keras_tokenizer.fit_on_texts(self.__train_data + self.__dev_data)
        self.__tokenized_train = self.__keras_tokenizer.texts_to_sequences(self.__train_data)
        self.__tokenized_dev = self.__keras_tokenizer.texts_to_sequences(self.__dev_data)
        self.__tokenized_test = self.__keras_tokenizer.texts_to_sequences(self.__test_data)

    def __padder(self):
        """Padding the test and train datasets with zeros to obtain same length vectors"""
        self.__tokenized_train = pad_sequences(self.__tokenized_train, padding="post", maxlen=MAXLEN)
        self.__tokenized_dev = pad_sequences(self.__tokenized_dev, padding="post", maxlen=MAXLEN)
        self.__tokenized_test = pad_sequences(self.__tokenized_test, padding="post", maxlen=MAXLEN)

    def __glove_embeddings_creator(self, glove_file_path: str, dimension: int, keras_output):
        """Creating the embeddings for the input data (tokenized) based ont he GloVe method"""
        glove_embedding = np.zeros((len(keras_output) + 1, dimension))
        with open(glove_file_path, "r", encoding='utf-8') as glove_file:
            for line in glove_file:
                word_key, *vector = line.split()
                if word_key in keras_output:
                    word_id = keras_output[word_key]
                    glove_embedding[word_id] = np.array(vector, dtype=np.float32)
        return glove_embedding

    def __build_cnn(self, embedding_matrix):
        sequence_input = Input(shape=(MAXLEN,), dtype='int32')
        # TODO: try with default keras embedding (should be worse performance)
        embedding_layer = Embedding(len(self.__keras_tokenizer.word_index) + 1,
                                    DIMENSION,
                                    weights=[embedding_matrix],
                                    input_length=MAXLEN,
                                    trainable=False)
        embedded_sequences = embedding_layer(sequence_input)
        x = layers.Conv1D(128, 5, activation='relu')(embedded_sequences)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(10, activation='relu')(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        model = Model(sequence_input, x)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model

    def run(self):
        print(f"{str(datetime.now())}: Loading the humor and non humor data set ...")
        self.__jokes_path = self.__file_path_creator(JOKES_PATH)
        self.__non_jokes_path = self.__file_path_creator(YAHOO_PATH)

        jokes, jokes_labels = self.__data_loader(self.__jokes_path, True)
        non_jokes, non_jokes_labels = self.__data_loader(self.__non_jokes_path, False)
        self.__corpus_creator(jokes, non_jokes, jokes_labels, non_jokes_labels)

        print(f"{str(datetime.now())}: Splitting data ...")
        self.__train_test_divider(self.__corpus, self.__corpus_labels)
        print(f"{str(datetime.now())}: Tokenizing data ...")
        self.__keras_tokenize()
        self.__padder()

        print(f"{str(datetime.now())}: Creating GloVe embedding ...")
        self.__glove_path = self.__file_path_creator(GLOVE_PATH)
        glove_embedding = self.__glove_embeddings_creator(self.__glove_path, DIMENSION,
                                                          self.__keras_tokenizer.word_index)
        np.save('glove_embedding', glove_embedding)
        print(f"{str(datetime.now())}: Building / training CNN model ...")
        model = self.__build_cnn(glove_embedding)
        self.__cnn_history = model.fit(np.asarray(self.__tokenized_train), np.asarray(self.__train_label),
                                       epochs=TRAIN_EPOCH,
                                       validation_data=(np.asarray(self.__tokenized_dev), np.array(self.__dev_label)),
                                       batch_size=TRAIN_BATCHSIZE)
        print(f"{str(datetime.now())}: Predicting the model on the test data ...")
        self.__cnn_results = model.evaluate(np.asarray(self.__tokenized_test), np.asarray(self.__test_label),
                                                       batch_size=TEST_BATCHSIZE, verbose=1)
        self.__cnn_predictions = model.predict(np.asarray(self.__tokenized_test), batch_size=TEST_BATCHSIZE, verbose=1)
        print(f"{str(datetime.now())}: The accuracy of the model on the test data set is: {self.__cnn_results}")




        model.save('glove_cnn')
        print(f"{str(datetime.now())}: Done")
