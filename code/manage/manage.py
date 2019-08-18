import os
import sys
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC

from code.preprocessor import preprocessing
from code.data import getData
from code.models import correlation
from code.models.nb.train_nb import generateNB_data
from code.models.smo.train_smo import generateSMO_data

def transform_sentences(sentences, result_dic):
    for j in range(len(sentences)):
        words = sentences[j].split(" ")
        for i in range(len(words)):
            if(words[i] in result_dic):
                words[i] = result_dic[words[i]]
        sentences[j] = " ".join(words)
    return sentences

def divide_data(X,Y):
    #mover
    x_sub = list()
    x_neu = list()
    for i, label in enumerate(Y):
        if(label == 1):
            x_sub.append(X[i])
        else:
            x_neu.append(X[i])
    return x_sub, x_neu

def train(options, cross_validation=False,corpus=None, labels=None):
    if(options.toPreprocess):
        getData.getCorpus(preprocess=True)
        correlation.correlation_dictionary()
    if(not cross_validation):
        corpus, labels = getData.getCorpus()
    correl_dic = correlation.correlation_dictionary(cross_validation=True)
    if(options.toCorrelate):
        corpus = transform_sentences(corpus, correl_dic)
    x_sub, x_neu = divide_data(corpus, labels)
    generateNB_data(corpus, labels, x_sub, x_neu)
    generateSMO_data(corpus, labels, x_sub, x_neu)

def create_model(options, main_path):
    vectorizer = CountVectorizer()
    if(options.useNB):
        file_path = os.path.join(main_path, 'models','nb')
    else:
        file_path = os.path.join(main_path, 'models','smo')
    fit_words = np.loadtxt(os.path.join(file_path,'fit_words.txt'), delimiter='  ', dtype='str')
    vectorizer.fit(fit_words)
    X = scipy.sparse.load_npz(os.path.join(file_path, 'X.npz'))
    Y = np.load(os.path.join(file_path, 'Y.npy'), allow_pickle=True)
    if(options.useNB):
        clf = ComplementNB(alpha=1).fit(X,Y)
    else:
        clf = SVC(kernel='rbf', gamma='auto', C=30).fit(X,Y)
    return clf, vectorizer


def interactive(options, main_path):
    clf, vectorizer = create_model(options, main_path)
    while(True):
        sentence = input('Sentence (-1 to leave): ')
        if(sentence=='-1'):
            break
        sentence = [sentence]
        sentence = preprocessing.preprocess(sentence)
        sentence = vectorizer.transform(sentence)
        predicted = clf.predict(sentence)
        if(predicted[0] == 0):
            print('Objective')
        else:
            print('Subjective')
    return
def file_classify(options, main_path):
    input_data = pd.read_csv(options.input, header=None, sep='\n')
    input_data = input_data.values
    clf, vectorizer = create_model(options, main_path)
    input_data = [sentence[0] for sentence in input_data]
    input_data = preprocessing.preprocess(input_data)
    input_data = [vectorizer.transform(sentence) for sentence in input_data]
    predicted = [clf.predict(sentence) for sentence in input_data]
    print(predicted)
    return
