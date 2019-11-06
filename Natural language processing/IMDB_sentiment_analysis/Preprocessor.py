import pickle
import multiprocessing
import os
import re

from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.util import mark_negation
from joblib import Parallel, delayed
from sklearn.utils import shuffle
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument

import datetime as dt
import pandas as pd

NUM_CORES = multiprocessing.cpu_count() // 2


# suporting methods to pickle objects to file
def save_obj(obj, filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def drop_html_and_unicode(text: str) -> str:
    """
    Dropping html and unicode characters
    """
    # strip HTML tags
    text = BeautifulSoup(markup=text, features="lxml").text
    # strip unicode characters
    text = re.sub(pattern=r'(\\u[0-9A-Fa-f]+)', repl=r'', string=text)
    text = re.sub(pattern=r'[^\x00-\x7f]', repl=r'', string=text)
    # strip urls
    text = re.sub(pattern=r'((www\.[A-Za-z0-9_\-\.]+))', repl=r'url',
                  string=text, )
    text = re.sub(pattern=r'((https?://[A-Za-z0-9_\-\.]+))', repl=r'url',
                  string=text, )
    return text
def replace_repetitions(text: str) -> str:
    """
    replacing repetitions of letters and special characters
    """
    # replace any repetition longer than 2 with exactly 2 occurences
    text = re.sub(pattern=r"(.)\1{2,}", repl=r"\1\1", string=text)
    # replace repetitions of special characters with a single occurence
    text = re.sub(pattern=r"([\!,\?\. ])\1{1,}", repl=r"\1", string=text)
    return text
def replace_symbolic_chars(text: str) -> str:
    """
    replacing special meaning of alone & and |
    """
    # in fact this step is only necessary for some approaches. It is not
    # if stop words will be removed in one of the following steps.
    text = re.sub(pattern=r"([ ,]\&[ ,])", repl=" and ",
                  string=text)  # assume there are spaces around &
    text = re.sub(pattern=r"([ ,]\|[ ,])", repl=" or ",
                  string=text)  # assume there are spaces around pipe
    return text
def replace_contractions(text: str) -> str:
    contractions_replacement_dict = {
        # contractions with not were identified from documents
        # some rare contractions or mistakes were skipped
        r"don't": " do not ", r" doesn't ": " does not ",
        r" didn't ": " did not ",
        r"can't": " can not ", r"isn't": " is not ", r"wasn't": " was not ",
        r"couldn't": " could not ", r"won't": " will not ",
        r"wouldn't": " would not ",
        r"aren't": " are not ", r"haven't": " have not ",
        r"weren't": " were not ",
        r"hasn't": " has not ", r"shouldn't": " should not ",
        r"hadn't": " had not ",
        r"ain't": " it is not ", r"needn't": " need not ",
        r"mustn't": " must not ",
        r"dosen't": " does not ", r"shan't": " shall not ",
        r"wan't": " was not ",
        r"Weren't": " were not ", r"Musn't": " must not ",
        r"arn't": " are not ",
        r"havn't": " have not ", r"Doesen't": " does not ",
        r"an't": " are not ",
        r"Idon't": " I do not ", r"Shudn't": " should not ",
        r"dosn't": " does not ",
        r"doeesn't": " does not ", r"heaven't": " have not ",
        r"doen't": " does not ",
        r"din't": " did not ", r"wqasn't": " was not ",
        r"mightn't": " might not ",
        r"hasen't": " has not ", r"shoudln't": " should not ",
        r"wern't": " were not ",
        r"wouln't": " would not ", r"doesen't": " does not ",
        # other contractions
        r"i'm": " i am ", r"(\w+)\'ll": "\g<1> will ",
        r"(\w+)\'ve": "\g<1> have ",
        r"(\w+)\'s": "\g<1> is ", r"(\w+)\'re": "\g<1> are ",
        r"(\w+)\'d": "\g<1> would "
    }
    # replaced contraction is surrounded with spaces to correct
    # contraction joined with preceding word
    for p in contractions_replacement_dict.keys():
        text = re.sub(pattern=p, repl=contractions_replacement_dict[p],
                      string=text, flags=re.IGNORECASE)
    # duplicated spaces removal
    text = re.sub(pattern="( )+", repl=" ", string=text)
    return text
def drop_numbers(text: str) -> str:
    # any integer number is wiped off
    return ''.join([x for x in text if not x.isdigit()])

def drop_special_chars(text: str) -> str:
    # some punctuation chars, used by mark_stopwords are not removed by
    # this method to avoid
    # negative influence on negation scope detection.
    # The mark_negation method limits scope to one of the following: '^[
    # .:;!?]$' .
    # So they remain untouched'
    text = re.sub(pattern=r"[^\w^ .:;!?]", repl="", string=text)
    text = re.sub(pattern="( )+", repl=" ", string=text)
    return text

def remove_stopwords(s: str) -> str:
    # stopword list were inherited from NLTK.
    # words important from the poin of view of negation detection were
    # eliminated
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours',
                 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
                 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                 'his', 'himself', 'she', "she's", 'her', 'hers',
                 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
                 'their', 'theirs', 'themselves', 'what',
                 'which', 'who', 'whom', 'this', 'that', "that'll", 'these',
                 'those',
                 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                 'have', 'has', 'had', 'having',
                 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
                 'but', 'if', 'or', 'because', 'as',
                 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                 'against', 'between', 'into',
                 'through', 'during', 'before', 'after', 'above', 'below',
                 'to', 'from', 'up', 'down', 'in',
                 'out', 'on', 'off', 'over', 'under', 'again', 'further',
                 'then', 'once', 'here', 'there',
                 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                 'each', 'few', 'more',
                 'most', 'other', 'some', 'such', 'nor', 'own', 'same',
                 'so', 'than', 'very', 's', 't',
                 'can', 'will', 'just', 'should', "should've", 'now', 'd',
                 'll', 'm', 'o', 're', 've', 'y',
                 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                 "didn't", 'doesn', "doesn't",
                 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
                 'isn', "isn't", 'ma', 'mightn',
                 "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
                 "shan't", 'shouldn', "shouldn't",
                 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
                 'wouldn', "wouldn't"]
    return ' '.join(
        [w for w in word_tokenize(s) if w.lower() not in stopwords])
def perform_stemming(text: str) -> str:
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(s) for s in word_tokenize(text)])

def perform_lemmatization(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(s) for s in word_tokenize(text)])

def tokenize_with_negations(text: str) -> list:
    # marking negative words and finally eliminating remaining punctuation
    # what is important - this method should be used as last.
    # it returns text tokenized
    # Surrounding stop mark with spaces is also important to get rid of
    # problem with
    # stop mark connected with the following word.
    text = mark_negation([x for x in word_tokenize(
        re.sub(pattern=r"\.", repl=" . ", string=text))])
    text = [x for x in text if
            x not in ['.', ':', ';', '!', '?', 'no', 'not']]
    return text

def perform_text_cleanup(text: str) -> str:
    text = drop_html_and_unicode(text)
    text = replace_repetitions(text)
    text = replace_symbolic_chars(text)
    text = drop_numbers(text)
    text = replace_contractions(text)
    text = perform_lemmatization(text)
    text = drop_special_chars(text)
    text = text.lower()
    return text

def perform_text_cleanup_limited(text: str) -> str:
    text = drop_html_and_unicode(text)
    text = replace_repetitions(text)
    # text = replace_symbolic_chars(text)
    # text = drop_numbers(text)
    # text = replace_contractions(text)
    # text = perform_lemmatization(text)
    # text = drop_special_chars(text)
    text = text.lower()
    return text


class Preprocessor(object):
    def __init__(self, ):
        self.dirs = {}
        self.texts = {}
        self.texts_after_cleanup = {}

    def load_raw_texts_from_dir(self, basedir: str):
        self.dirs[('train', 'pos')] = basedir + "train//pos//"
        self.dirs[('train', 'neg')] = basedir + "train//neg//"
        self.dirs[('test', 'pos')] = basedir + "//test//pos//"
        self.dirs[('test', 'neg')] = basedir + "//test//neg//"

        # loading texts to list stored in a single dictionary
        for d in self.dirs:
            self.texts[d] = []
            for t in os.listdir(self.dirs[d]):
                f = open(file=self.dirs[d] + t, mode="r", encoding="utf-8")
                self.texts[d].append(f.read())
        print("Datasets loaded")
        for d in self.texts:
            print('Dataset: {}:{}, number of texts: {}'.format(d[0], d[1], len(
                self.texts[d])))

    def pickle_raw_texts(self, pickle_filename: str):
        save_obj(self.texts, pickle_filename)

    def load_raw_texts_from_pickle(self, pickle_filename):
        self.texts = load_obj(pickle_filename)
        print("Datasets loaded")
        for d in self.texts:
            print('Dataset: {}:{}, number of texts: {}'.format(d[0], d[1], len(
                self.texts[d])))

    def perform_text_cleanup(self):
        start=dt.datetime.now()
        for d in self.texts.keys():
            self.texts_after_cleanup[d] = Parallel(n_jobs=NUM_CORES)(
                delayed(perform_text_cleanup)(s) for s in self.texts[d])
        print('Texts cleanup finished in {}'.format(dt.datetime.now() - start))

    def perform_text_cleanup_limited(self):
        start=dt.datetime.now()
        for d in self.texts.keys():
            self.texts_after_cleanup[d] = Parallel(n_jobs=NUM_CORES)(
                delayed(perform_text_cleanup)(s) for s in self.texts[d])
        print('Texts cleanup finished in {}'.format(dt.datetime.now() - start))

    def build_datasets(self,train_eval_ratio=1):
        # train data preparation
        train_data_pos = pd.DataFrame(self.texts_after_cleanup[('train', 'pos')])
        train_data_pos['Sentiment'] = 1
        train_data_neg = pd.DataFrame(self.texts_after_cleanup[('train', 'neg')])
        train_data_neg['Sentiment'] = 0
        train_data = train_data_pos.append(train_data_neg)
        train_data.reset_index(inplace=True, drop=True)
        train_data.columns = ['Review', 'Sentiment']
        self.train_data = shuffle(train_data)

        # test data preparation
        test_data_pos = self.texts_after_cleanup[('test', 'pos')]
        test_data_neg = self.texts_after_cleanup[('test', 'neg')]
        test_data = pd.DataFrame(
            [(s, 1) for s in test_data_pos] + [(s, 0) for s in test_data_neg])
        test_data.columns = ['Review', 'Sentiment']
        self.test_data = shuffle(test_data)

        print(
            "Storing reviews in two dataframes with shapes: train_data: {}, test_data: {}" \
            .format(self.train_data.shape, self.test_data.shape))

    def save_datasets(self, dirname:str):
        self.train_data.to_pickle(dirname+'train_data.pkl')
        self.test_data.to_pickle(dirname + 'test_data.pkl')

    def load_datasets(self, dirname:str):
        self.train_data=pd.read_pickle(dirname+'train_data.pkl')
        self.test_data=pd.read_pickle(dirname + 'test_data.pkl')
        print('Data loaded form :',dirname)

    def build_train_corpus(self):
        all_tokenized = [simple_preprocess(x) for x in self.train_data['Review']]
        all_tokenized.extend(simple_preprocess(x) for x in self.test_data['Review'])
        self.train_corpus = [TaggedDocument(x, [i]) for i, x in
                        enumerate(all_tokenized)]

    def save_train_corpus(self, path:str):
        save_obj(self.train_corpus, path+'train_corpus.pkl')

    def load_train_corpus(self, path:str):
        self.train_corpus=load_obj(path+'train_corpus.pkl')