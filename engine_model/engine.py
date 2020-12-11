from datetime import datetime
from typing import List
import nltk
import pandas as pd
import seaborn
import os
#import copy
pd.set_option('max_colwidth', None)
from nltk import word_tokenize, pos_tag
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

from nltk import word_tokenize
from nltk.tag import pos_tag


from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn import metrics


from sklearn.metrics import confusion_matrix

from sklearn.svm import LinearSVC
from tensorflow.keras import layers
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Embedding, concatenate
from tensorflow.python.keras.models import Sequential, Model

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import metrics
ORIGINAL_DATA = "original_corpus.npy"
PROCESSED_DATA = "processed_corpus.npy"
LABELS = "corpus_labels.npy"
GLOVE_PATH = "glove/glove.6B.100d.txt"
MAXLEN = 50
DIMENSION = 100
TRAIN_EPOCH = 20
TRAIN_BATCHSIZE = 256
TEST_BATCHSIZE = 16
USE_CUSTOM_FEATURES = True
USE_PREPROCESSING = True
NUM_FEATURES = 2
XLABEL = "Epoch"
ACCURACYPLOT = "Acurracy_Plot"
LOSSPLOT = "Loss_Plot"
FIRST_PERSON_WORDS = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
SECOND_PERSON_WORDS = ['you', 'your', 'yours']
LOWER_LIMIT = 0.35
UPPER_LIMIT = 0.65
#nlp = spacy.load("en_core_web_sm")
#nlp = spacy.load("en")
# import spacy
# nlp = spacy.en_core_web_sm.load()


class Classifier():

    def __init__(self):
        self.__keras_tokenizer = Tokenizer(num_words=35000, filters='', lower=False)
        self.__vectorizer = CountVectorizer()
        self.__TfidfTransformer = TfidfTransformer(use_idf=True)
        self.__vanilla_classifier = CalibratedClassifierCV(
            LinearSVC(class_weight='balanced', penalty='l2', loss='squared_hinge', dual=True), method='isotonic')
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
        self.__train_features = None
        self.__dev_features = None
        self.__test_features = None

        # Vanilla classifier attributes
        self.__vanilla_predict_result = None
        self.__vanilla_accuracy = True
        self.__vanilla_probs = None
        self.__vanilla_unconfident = None

        self.__glove_embedding = None
        self.__jokes_path = None
        self.__non_jokes_path = None
        self.__base_path = None
        self.__glove_path = None
        self.__cnn_history = None
        self.__cnn_results = None
        self.__cnn_predictions = None
        self.__base_path = None

    def __directory_transfer(self, path:str):
        """Change the directory adress to the desired adress"""
        self.__base_path = os.getcwd()
        print(f"The base path is {self.__base_path}")
        os.chdir(self.__base_path+"/"+path)
        print(f"The path has been changed to {os.getcwd()} ...")


    def __train_test_divider(self, data, data_label):
        """Splitting the data between train, dev and test
        as the input it can take the original corpus or the processed corpus
        the original corpus or the processed corpus all are of type numpy.ndarray"""
        self.__train_data, self.__test_data, self.__train_label, self.__test_label = \
            train_test_split(data, data_label, test_size=0.1, random_state=1000)

        self.__train_data, self.__dev_data, self.__train_label, self.__dev_label = \
            train_test_split(self.__train_data, self.__train_label, test_size=0.15 / (0.85 + 0.15), random_state=1000)

    def __corpus_embedding_creator(self, input_data, original_corpus: bool):
        """create the bag-of-words representation from the input corpus
        In the case of using the train, dev and test set from original corpus,
         the corpus should be vectorized ..."""
        if original_corpus:
            input_data = self.__vectorizer.fit_transform(input_data)
        return self.__TfidfTransformer.fit_transform(input_data)

    def __build_vanilla_classifier(self):
        """Build a weighted classifier as the vanilla classifier"""
        self.__vanilla_classifier.fit(self.__train_data, self.__train_label)

    def __keras_tokenize(self):
        """Tokenize the train and test datasets""" #TODO probably the tokenizer should be trained on the test as well
        self.__keras_tokenizer.fit_on_texts(self.__train_data + self.__dev_data)
        self.__tokenized_train = self.__keras_tokenizer.texts_to_sequences(self.__train_data)
        self.__tokenized_dev = self.__keras_tokenizer.texts_to_sequences(self.__dev_data)
        self.__tokenized_test = self.__keras_tokenizer.texts_to_sequences(self.__test_data)

    def __padder(self):
        """Padding dat sets with zeros to obtain same length vectors"""
        self.__tokenized_train = pad_sequences(self.__tokenized_train, padding="post", maxlen=MAXLEN)
        self.__tokenized_dev = pad_sequences(self.__tokenized_dev, padding="post", maxlen=MAXLEN)
        self.__tokenized_test = pad_sequences(self.__tokenized_test, padding="post", maxlen=MAXLEN)


    def __create_feature_array(self, data, count = NUM_FEATURES):
        """Adding hand-crafted features to the corpus"""
        features = np.empty(shape=(len(data), count))
        for i, sentence in enumerate(data):
            sentence_lower = sentence.lower()
            words = word_tokenize(sentence_lower)
            sentence_length = len(words)
            features[i, 0] = len([w for w in words if w in FIRST_PERSON_WORDS]) / sentence_length
            features[i, 1] = len([w for w in words if w in SECOND_PERSON_WORDS]) / sentence_length
            # if count > 2:
            #     # Third component of the custom feature
            #     # Proper nouns
            #     print(f"Evaluating the proper nouns within the corpus ... ")
            #     pos_tags = pos_tag(word_tokenize(sentence))
            #     proper_nouns = [word for word, pos in pos_tags if pos in ['NNP', 'NNPS']]
            #     features[i, 2] = len(proper_nouns) / sentence_length
            #
            #     #NER for people
            #     doc = nlp(sentence)
            #     person_ents = [(X.text, X.label_) for X in doc.ents if X.label_ == 'PERSON']
            #     features[i, 2] = len(person_ents)
        return features
    def __create_feature_wrapper(self):
        """Wrapepr function for the function __create_feature_array"""
        self.__train_features = self.__create_feature_array(self.__train_data)
        self.__dev_features = self.__create_feature_array(self.__dev_data)
        self.__test_features = self.__create_feature_array(self.__test_data)

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

    def __build_cnn(self, embedding_matrix, costume_feature_flag: bool, count = NUM_FEATURES):
        """ Build the CNN model """
        sequence_input = Input(shape=(MAXLEN,), dtype='int32', name='Sequence')
        embedding_layer = Embedding(len(self.__keras_tokenizer.word_index) + 1,
                              DIMENSION,
                              weights=[embedding_matrix],
                              input_length=MAXLEN,
                              trainable=False)
        embedded_sequences = embedding_layer(sequence_input)
        x = layers.Dropout(0.1)(embedded_sequences)
        x = layers.Conv1D(128, 3, activation='relu')(x)
        x = layers.MaxPooling1D(3, padding='same')(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(3, padding='same')(x)
        x = layers.Conv1D(128, 2, activation='relu', padding='same')(x)
        x = layers.GlobalMaxPooling1D()(x)
        if costume_feature_flag:
            features = Input(shape=(count,), dtype='float32', name='Features')
            merged = layers.Concatenate()([x, features])
            merged = layers.Dropout(0.1)(merged)
            merged = layers.Dense(20, activation='relu')(merged)
            merged = layers.Dense(1, activation='sigmoid')(merged)
            model = Model([sequence_input, features], merged)
        else:
            merged = x
            merged = layers.Dropout(0.1)(merged)
            merged = layers.Dense(20, activation='relu')(merged)
            merged = layers.Dense(1, activation='sigmoid')(merged)
            model = Model(sequence_input, merged)

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model
    def __confusion_matrix_builder(self, lebels, predicitions, method_name:str):
        """Create a confusion matrix based on the ground truth labels and predicitions"""
        cm = confusion_matrix(lebels, predicitions)
        plot_cm = pd.DataFrame(cm, index=[i for i in ["Humor", "Non_Humor"]],
                               columns=[i for i in ["Humor", "Non_Humor"]])
        plt.figure(figsize=(10, 7))
        plt.title(f"Confusion Matrix for the method {method_name}")
        seaborn.heatmap(plot_cm, annot=True)
        # plt.show()

    def __convergance_plot_builder(self, history):
        """Create the convergence plots for th loss and accuracy of the model"""
        self.__directory_transfer("Results")
        labels = [key for key in history.history.keys()]
        print(f"Creating the accuracy plot for the model's training history")
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        print(labels)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel(XLABEL)
        plt.legend(["training_set", "validation_set"], loc="upper left")
        # plt.show()
        plt.savefig(ACCURACYPLOT)

        plt.subplot(1, 2, 2)
        print(f"Creating the loss plot for the model's training history")
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel(XLABEL)
        plt.legend(["training_set", "validation_set"], loc="upper left")
        # plt.show()
        plt.savefig(self.__base_path + LOSSPLOT)

    def run(self, processed_data: bool, vanilla_classifier: bool, custom_features: bool):
        self.__directory_transfer("Results")
        print(f"{str(datetime.now())}: Loading the humor and non humor data set to create the corpus ...")
        if processed_data:
            print(f"For this experiment the pre-processed corpus is being used ... ")
            self.__corpus = np.load(self.__base_path+"/"+"Results"+PROCESSED_DATA)
        else:
            print(f"For this experiment the original corpus without any pre-processing is being used ...")
            self.__corpus = np.load(self.__base_path+"/"+"Results"+ORIGINAL_DATA)
        corpus_lables = np.load(self.__base_path+"/"+"Results"+LABELS)

        if vanilla_classifier:
            print(f"In this experiment the used classifier is a vanilla classifier ...")
            print(f"{str(datetime.now())}: Create bag of words representation of the corpus...")
            if processed_data:
                self.__corpus = self.__corpus_embedding_creator(self.__corpus, original_corpus=False)
            else:
                self.__corpus = self.__corpus_embedding_creator(self.__corpus, original_corpus=True)

            print(f"{str(datetime.now()): Dividing the used corpus embeddings to train, dev and test sets ...}")
            self.__train_test_divider(self.__corpus, corpus_lables)
            print(f"{str(datetime.now())}: Creating the weighted classifier as the vanilla base classifier...")
            self.__build_vanilla_classifier()
            self.__vanilla_predict_result = self.__vanilla_classifier.predict(self.__dev_data)
            self.__confusion_matrix_builder(self.__dev_label, self.__vanilla_predict_result,
                                            method_name="Vanila_Claasifier")
            self.__vanilla_accuracy = metrics.accuracy_score(self.__dev_label, self.__vanilla_predict_result)
            print('Test accuracy : ' + str('{:04.2f}'.format(self.__vanilla_accuracy * 100)) + ' %')
            print('Report on Test_set \n', classification_report(self.__vanilla_predict_result, self.__dev_label))

            self.__vanilla_probs = self.__vanilla_classifier.predict_proba(self.__dev_data)
            self.__vanilla_unconfident = [(i, row) for i, row in enumerate(self.__vanilla_probs)
                                          if LOWER_LIMIT < row[0] < UPPER_LIMIT]


        print(f"In this experiment the used classifier is a deep learning model ...")
        if custom_features: #TODO Put the function befor the preprocessing
            print(f"{str(datetime.now())}: Creating feature arrays ...")
            self.__create_feature_arrays()
        #
        # print(f"{str(datetime.now())}: Tokenizing data ...")
        # self.__keras_tokenize()
        # self.__padder()
        #
        # print(f"{str(datetime.now())}: Creating GloVe embedding ...")
        # self.__glove_path = self.__file_path_creator(GLOVE_PATH)
        # glove_embedding = self.__glove_embeddings_creator(self.__glove_path, DIMENSION,
        #                                                   self.__keras_tokenizer.word_index)
        # np.save('glove_embedding', glove_embedding)
        # if vanilla_classifier:
        #     print(f"{str(datetime.now())}: Training the vanilla classifier ...")
        #     self.__build_vanilla_classifier()
        #

        # print(f"{str(datetime.now())}: Building / training CNN model ...")
        # model = self.__build_cnn(glove_embedding)
        # print(model.summary())
        # if custom_features:
        #     train_in = [self.__tokenized_train, self.__train_features]
        #     dev_in = [self.__tokenized_dev, self.__dev_features]
        #     test_in = [np.asarray(self.__tokenized_test), np.asarray(self.__test_features)]
        # else:
        #     train_in = self.__tokenized_train
        #     dev_in = self.__tokenized_dev
        #     test_in = self.__tokenized_test
        #
        # self.__cnn_history = model.fit(train_in,
        #                                np.asarray(self.__train_label),
        #                                epochs=TRAIN_EPOCH,
        #                                validation_data=(dev_in, np.array(self.__dev_label)),
        #                                batch_size=TRAIN_BATCHSIZE)
        # self.__convergance_plot_builder(self.__cnn_history)
        # print(f"{str(datetime.now())}: Predicting the model on the test data ...")
        # self.__cnn_results = model.evaluate(test_in,
        #                                     np.asarray(self.__test_label),
        #                                     batch_size=TEST_BATCHSIZE, verbose=1)
        # self.__cnn_predictions = model.predict(test_in,
        #                                        batch_size=TEST_BATCHSIZE, verbose=1)
        # print(f"{str(datetime.now())}: The accuracy of the model on the test data set is: {self.__cnn_results}")
        #
        # self.__cnn_predictions = model.predict(dev_in, batch_size=TEST_BATCHSIZE, verbose=1)
        # # print(self.__cnn_predictions)
        # accuracy_score_dev = metrics.accuracy_score(self.__dev_label, self.__cnn_predictions.round())
        # print('Dev accuracy : ' + str('{:04.2f}'.format(accuracy_score_dev)) + ' %')
        # print('Report on Dev_set \n', classification_report(self.__cnn_predictions.round(), self.__dev_label))
        #
        # model.save('glove_cnn')
        # print(f"{str(datetime.now())}: Done")





