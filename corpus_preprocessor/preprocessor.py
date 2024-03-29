import string
import os
import csv
from typing import List
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag
from datetime import datetime
import numpy as np

# nlp = spacy.load("en_core_web_sm")
# nlp = spacy.load("en")
# import spacy
# nlp = spacy.en_core_web_sm.load()

JOKES = "jokes_processed_20201110.csv"
NO_JOKES = "no_jokes_amaz_yahoo_20201204.csv"
HUMOR_PATH = "humor_dataset.csv"
RESULT = "Results"

FIRST_PERSON_WORDS = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
SECOND_PERSON_WORDS = ['you', 'your', 'yours']

class TextProcessor():

    def __init__(self):
        self.__main_corpus = []
        self.__corpus_labels = []
        self.__train_data = []
        self.__dev_data = []
        self.__test_data = []
        self.__train_label = []
        self.__dev_label = []
        self.__test_label = []
        self.__word_tokenize = word_tokenize
        self.__lemmatizer = WordNetLemmatizer
        self.__tagger = pos_tag
        self.__stemmer = PorterStemmer
        self.__stop_words = set(stopwords.words('english'))
        self.__punctuation = string.punctuation

        # self.__humourous_words = {}
        # with open(self.__file_path_creator(HUMOR_PATH), encoding="utf-8") as csvfile:
        #     readCSV = csv.reader(csvfile, delimiter=',')
        #     next(readCSV, None)
        #     for row in readCSV:
        #         self.__humourous_words[row[0]] = float(row[1])

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
                    df.append(row[1])
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
        self.__corpus_labels = non_joke_labels

    def __pre_process_text(self, sentence: str, lemmatize: bool, stemming: bool, stop_words: bool, punctuation: bool):
        """Pre-process a sentec in the corpus. Pre-processing includes:
        1. lower case the letters
        2. replacement and
        3. Lemmatizing
        4. Stemming
        5, Ommit punctuation
        6. Ommit stop words
        Between the stemming and lemmatizing only one flag should be turned on.
        """
        print(f"Transforming the letters to lower case ....")
        sentence = sentence.lower()
        print(f"Replacing short version of expressions with their formal version ...")
        replace_dict = {"don't": "do not", "can't": "can not", "doesn't": "does not", "haven't": "have not",
                        "hasn't": "has not", "wouldn't": "would not", "won't": "will not", "didn't": "did not", "wasn't": "was not",
                        "weren't": "were not", "it's": "it is", "what's": "what is", "you're": "you are",
                        "they're": "they are", "we're": "we are"}
        for key in replace_dict:
            sentence.replace(key, replace_dict[key])
        print(f"Tokenizing the input sentence ...")
        sentence_token = self.__word_tokenize(sentence)
        if lemmatize:
            print(f"Lematizing the input sentence ...")
            tag_map = defaultdict(lambda: wn.NOUN)
            tag_map['J'] = wn.ADJ
            tag_map['V'] = wn.VERB
            tag_map['R'] = wn.ADV
            sentence_token = [self.__lemmatizer.lemmatize(token, tag_map[tag[0]]) for token, tag in
                                  self.__tagger(sentence_token)]
        if stemming:
            print(f"Stemming the input sentence...")
            sentence_token = [self.__stemmer().stem(token) for token in sentence_token]
        if stop_words:
            print(f"Eliminating the stop words from the input sentence ...")
            sentence_token = [token for token in sentence_token
                              if token not in self.__stop_words]
        if punctuation:
            print(f"Eliminating the punctuation from the input sentence ... ")
            sentence_token = [word for word in sentence_token
                              if word not in self.__punctuation]
        return ' '.join([word for word in sentence_token])

    def __pre_process_wrapper(self, input_data: List[str], *args):
        """Wrapper function for the function __pre_process_text"""
        return [self.__pre_process_text(sentence, *args) for sentence in input_data]

    def __create_feature_array(self, input_data: List[str]):
        """Adding hand-crafted features to the corpus"""
        features = np.empty(shape=(len(input_data), 3))
        for i, sentence in enumerate(input_data):
            sentence_lower = sentence.lower()
            words = word_tokenize(sentence_lower)
            sentence_length = len(words)
            features[i, 0] = len([w for w in words if w in FIRST_PERSON_WORDS]) / sentence_length
            features[i, 1] = len([w for w in words if w in SECOND_PERSON_WORDS]) / sentence_length
            # if count > 2:
            #     # Third component of the custom feature
            #     # Proper nouns
            print(f"Evaluating the proper nouns within the corpus ... ")
            pos_tags = pos_tag(word_tokenize(sentence))
            proper_nouns = [word for word, pos in pos_tags if pos in ['NNP', 'NNPS']]
            features[i, 2] = len(proper_nouns) / sentence_length
            #     #NER for people
            #     doc = nlp(sentence)
            #     person_ents = [(X.text, X.label_) for X in doc.ents if X.label_ == 'PERSON']
            #     features[i, 2] = len(person_ents)

            ## Humourous words
            # humour_rating = [self.__humourous_words.get(w, 0) for w in words]
            # features[i, 3] = max(humour_rating)
        np.save("features", features)

    def run(self):
        print(f"{str(datetime.now())}: Loading the humor and non humor data set ...")
        self.__jokes_path = self.__file_path_creator(JOKES)
        self.__non_jokes_path = self.__file_path_creator(NO_JOKES)
        self.base_path = self.__jokes_path[:-len(JOKES)]
        try:
                os.mkdir(self.base_path+"/"+RESULT)
        except:
            print(f"The desired directory already exists...")
        else:
            print(f"The directory is successfully created ...")

        jokes, jokes_labels = self.__data_loader(self.__jokes_path, True)
        non_jokes, non_jokes_labels = self.__data_loader(self.__non_jokes_path, False)
        print(f"Creating the main dataset consisting of humor and non-humor ...")
        self.__corpus_creator(jokes, non_jokes, jokes_labels, non_jokes_labels)
        os.chdir(self.base_path+"/"+RESULT)
        # np.savez_compressed("original_corpus", self.__main_corpus)

        print(f" Pre-processing the corpus ...")
        print(f"Creating the custom features array...")
        self.__create_feature_array(self.__main_corpus)

        print(f" Pre-processing the corpus ...")
        print(f"first phase without lemmatizing or stemming ....")
        print(f"Eliminating both stop words and punctuations ...")
        processed_corpus = self.__pre_process_wrapper(self.__main_corpus, False, False, True, True)
        np.savez_compressed("sw_punct", processed_corpus)

        # print(f"Only eliminating the stop words ...")
        # processed_corpus = self.__pre_process_wrapper(self.__main_corpus, False, False, True, False)
        # np.savez_compressed("sw", processed_corpus)
        # print(f"Only eliminating the punctuations ... ")
        # processed_corpus = self.__pre_process_wrapper(self.__main_corpus, False, False, False, True)
        # np.savez_compressed("punct", processed_corpus)

        # print(f"Second phase without only lemmatizing ...")
        # print(f"Eliminating both stop words and punctuations ...")
        # processed_corpus = self.__pre_process_wrapper(self.__main_corpus, True, False, True, True)
        # np.savez_compressed("lemma_sw_punct", processed_corpus)
        # print(f"Only eliminating the stop words ...")
        # processed_corpus = self.__pre_process_wrapper(self.__main_corpus, True, False, True, False)
        # np.savez_compressed("lemma_sw", processed_corpus)
        # print(f"Only eliminating the punctuations ... ")
        # processed_corpus = self.__pre_process_wrapper(self.__main_corpus, True, False, False, True)
        # np.savez_compressed("lemma_punct", processed_corpus)
        # print(f"Now only doing the lemmatization ...")
        # processed_corpus = self.__pre_process_wrapper(self.__main_corpus, True, False, False, False)
        # np.savez_compressed("lemma", processed_corpus)

        # print(f"Last phase without stemming ....")
        # print(f"Eliminating both stop words and punctuations ...")
        # processed_corpus = self.__pre_process_wrapper(self.__main_corpus, False, True, True, True)
        # np.savez_compressed("stemm_sw_punct", processed_corpus)
        # print(f"Only eliminating the stop words ...")
        # processed_corpus = self.__pre_process_wrapper(self.__main_corpus, False, True, True, False)
        # np.savez_compressed("stemm_sw", processed_corpus)
        # print(f"Only eliminating the punctuations ... ")
        # processed_corpus = self.__pre_process_wrapper(self.__main_corpus, False, True, False, True)

        # np.savez_compressed("stemm_punct", processed_corpus)
        # print(f"Now only doing the stemming ....")
        # processed_corpus = self.__pre_process_wrapper(self.__main_corpus, False, True, False, False)
        # np.savez_compressed("stemm", processed_corpus)

        print(f"Saving the labels of the corpus")
        np.save("corpus_labels", self.__corpus_labels)



