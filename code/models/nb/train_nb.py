import numpy as np
import os
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer

'''Recebe vetor com o numero de ocorrencias de cada palavra em cada teste.
calcula as frequencias e gera o vetor apenas com os necessarios
'''
def word_freq(x):
    vec = CountVectorizer().fit(x)
    bag_of_words = vec.transform(x)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:]

def select_words(x_sub, x_neu, number):
    sel_sub = list()
    sel_neu = list()
    for i in range(number):
        sel_sub.append(x_sub[i][0])
        sel_neu.append(x_neu[i][0])
   # sel_sub = x_sub[0:number][1]
   # sel_neu = x_neu[0:number][1]
    words = [x for x in sel_sub if x not in sel_neu]
    for x in sel_neu:
        if x not in sel_sub:
            words.append(x)
   # words = [x for x in sel_neu if x not in sel_sub]
    return words

def generateNB_data(X, Y, x_sub, x_neu):
    x_sub = word_freq(x_sub)
    x_neu = word_freq(x_neu)
    fit_words = select_words(x_sub, x_neu, number = 99)
    vectorizer = CountVectorizer()
    vectorizer.fit(fit_words)
    X = vectorizer.transform(X)
    #save X and Y
    my_path = os.path.split(os.path.abspath(__file__))[0]
    scipy.sparse.save_npz(os.path.join(my_path,'X.npz'), X)
    Y = np.array(Y)
    np.save(os.path.join(my_path,'Y'), Y)
    np.savetxt(os.path.join(my_path,'fit_words.txt'), fit_words, delimiter="  ", fmt="%s")
    #sparse_matrix = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')
