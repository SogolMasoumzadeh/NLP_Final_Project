from sentence_transformers import SentenceTransformer
from typing import List
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tf.keras.preprocessing.text import Tokenizer
from tf.keras.preprocessing.sequence import pad_sequences
import numpy as np

JOKES_PATH = "/Users/sogolmsz/Documents/Projects/PhD./PhD. Courses/First term/NLP/Final_Project/jokes_processed_20201110.csv"
YAHOO_PATH = "/Users/sogolmsz/Documents/Projects/PhD./PhD. Courses/First term/NLP/Final_Project/df_yahoo_news_20201123.csv"
GLOVE_PATH = "/Users/sogolmsz/Documents/Projects/PhD./PhD. Courses/First term/NLP/Final_Project/glove/glove.6B.50d.txt"
MAXLEN = 100
DIEMNSION = 50

class classifer():

    def __init__(self):
        self.__spliter = train_test_split(test_size=0.25, random_state=1000)
        self.__keras_tokenizer = Tokenizer(num_words=5000)
        self.__corpus = []
        self.__corpus_labels = []
        self.__train_data = []
        self.__test_data = []
        self.__train_label = []
        self.__test_label = []
        self.__tokenized_train = []
        self.__tokenized_test = []
        self.__glove_embedding = None

    def __data_loader(self, file_path: str, joke_falg: bool):
        """Data loader from the .csv files"""
        df = []
        label = []

        if joke_falg:
            _file = pd.read_csv(file_path, names=["text", "text_fianl", "question_flag"])
            df.append(_file)
            label = np.ones(df[0].shape[0])

        if not joke_falg:
            _file = pd.read_csv(file_path, names=["text"])
            df.append(_file)
            label = np.zeros(df[0].shape[0])
        return df, label

    def __corpus_creator(self, joke_list, non_joke_list, joke_labels: List[int], non_joke_labels: List[int]):


        for joke in joke_list[0][0]:
            non_joke_list[0][0].append(joke)
        for label in joke_labels:
            non_joke_labels.append(label)
        self.__corpus = non_joke_list
        self.__corpus_labels = non_joke_labels


    def __train_test_devider(self, data_file: List[str], data_label):
        """Splitting the data between train and test"""
        self.__train_data, self.__test_data, self.__train_label, self.__test_label = self.__spliter(data_file, data_label)

    def __keras_tokenizer(self):
        """Tokenize the train and test datasets"""
        self.__keras_tokenizer.fit_on_texts(self.__train_data)
        self.__tokenized_train = self.__keras_tokenizer.texts_to_sequences(self.__train_data)
        self.__tokenized_test = self.__keras_tokenizer.texts_to_sequences(self.__test_data)

    def __padder(self):
        """Padding the test and train datasets with zeros to obtain same length vectors"""
        self.__tokenized_train = pad_sequences(self.__tokenized_train, padding="post", maxlen=MAXLEN)
        self.__tokenized_test = pad_sequences(self.__tokenized_test, padding="post", maxlen=MAXLEN)

    def __glove_embeddings_creator(self, glove_file_path: str, dimension: int, keras_output):
        """Creating the embeddings for the input data (tokenized) based ont he GloVe method"""
        glove_embedding = np.zeros((len(keras_output)+1, dimension))
        with open(glove_file_path, "r") as glove_file:
            for line in glove_file:
                word_key, *vector = line.split()
                if word_key in keras_output:
                    word_id = keras_output[word_key]
                    glove_embedding[word_id] = np.array(vector, dtype=np.float32)
        return glove_embedding












    def run(self):

        print(f"Loading the humor and non humor data set ...")
        jokes, jokes_labels = self.__data_loader(JOKES_PATH, True)
        non_jokes, non_jokes_labels = self.__data_loader(YAHOO_PATH, False)
        self.__corpus_creator(jokes, non_jokes)
        self.__train_test_devider(self.__corpus, self.__corpus_labels)
        self.__keras_tokenizer()
        self.__padder()
        train_glove_embedding = self.__glove_embeddings_creator(GLOVE_PATH, DIEMNSION, self.__tokenized_train)
        test_glove_embedding = self.__glove_embeddings_creator(GLOVE_PATH, DIEMNSION, self.__tokenized_test)



