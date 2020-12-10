import os
import csv
from typing import List
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from datetime import datetime
import numpy as np

JOKES = "jokes_processed_20201110.csv"
NO_JOKES = "no_jokes_amaz_yahoo_20201204.csv"

class TextProcessor():

    def __init__(self):
        self.__main_corpus = []
        self.__main_corpus_labels = []
        self.__train_data = []
        self.__dev_data = []
        self.__test_data = []
        self.__train_label = []
        self.__dev_label = []
        self.__test_label = []
        self.__word_tokenize = word_tokenize()
        self.__lemmatizer = WordNetLemmatizer()
        self.__tagger = pos_tag()

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
        """Combine jokes and non-jokes into single data set"""
        non_joke_list.extend(joke_list)
        non_joke_labels.extend(joke_labels)
        self.__main_corpus = non_joke_list
        self.__main_corpus_labels = non_joke_labels

    def __pre_process_text(self, sentence: str):
        """Pre-process a sentec in the corpus. Pre-processing includes:
        1. lower case the letters
        2. replacement and
        3. Lemmatizing
        """
        print(f"Transforming the letters to lower case ....")
        sentence = sentence.lower()
        print(f"Replacing short version of expressions with their formal version ...")
        replace_dict = {"don't": "do not", "can't": "can not", "doesn't": "does not", "haven't": "have not",
                        "hasn't": "has not", "wouldn't": "would not", "won't": "will not", "didn't": "did not", "wasn't": "was not",
                        "weren't": "were not", "it's": "it is", "what's": "what is", "you're": "you are",
                        "they're": "they are", "we're": "we are"}
        for key in replace_dict:
            sentence.replace(replace_dict[key])
        print(f"Tokenizing the input sentence ...")
        sentence_token = self.__word_tokenize(sentence)
        print(f"Lematizing the input sentence ...")
        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        lematized_sentence = [self.__lemmatizer.lemmatize(token, tag_map[tag[0]]) for token, tag in
                              self.__tagger(sentence_token)]
        return ' '.join([word for word in lematized_sentence])

    def __pre_process_wrapper(self, input_data: List[str]):
        """Wrapper function for the function __pre_process_text"""
        return [self.__pre_process_text(sentence) for sentence in input_data]

    def run(self):
        print(f"{str(datetime.now())}: Loading the humor and non humor data set ...")
        self.__jokes_path = self.__file_path_creator(JOKES)
        self.__non_jokes_path = self.__file_path_creator(NO_JOKES)
        self.__base_path = self.__jokes_path[:-len(JOKES)]

        jokes, jokes_labels = self.__data_loader(self.__jokes_path, True)
        non_jokes, non_jokes_labels = self.__data_loader(self.__non_jokes_path, False)
        print(f"Creating the main dataset consisting of humor and non-humor ...")
        self.__corpus_creator(jokes, non_jokes, jokes_labels, non_jokes_labels)
        np.save("original_corpus", self.__main_corpus)
        print(f" Pre-processing the corpus and lemmatize it ...")
        processed_corpus = self.__pre_process_wrapper(self.__main_corpus)
        np.save("processed_corpus", processed_corpus)


