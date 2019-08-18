import numpy as np
import os
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer

def word_freq(x):
    vec = CountVectorizer().fit(x)
    bag_of_words = vec.transform(x)
    sum_words = bag_of_words.sum(axis=0)
    total = sum_words.sum()
    words_freq = [(word, sum_words[0, idx], sum_words[0,idx]/total) for word, idx in     vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[0], reverse=False)
    return words_freq[:], total

#total é o numero de palavras na classe, e a freq = sum_words/total
#sum_words sera ainda utilizado, portanto o total é passado
def calculate_cmfs(x_sub,x_neu, total):
    smo_sub = list()
    smo_neu = list()
    for tup in x_sub:
        prob_sec = 1 #P(ck/wk)
        for tup_find in x_neu:
            if(tup_find[0]==tup[0]):
                prob_sec = tup[1]/(tup[1]+tup_find[1])
                break
        smo = tup[2]*prob_sec
        smo_sub.append((tup[0], smo))

    for tup in x_neu:
        prob_sec = 1 #P(ck/wk)
        for tup_find in x_sub:
            if(tup_find[0]==tup[0]):
                prob_sec = tup[1]/(tup[1]+tup_find[1])
                break
        smo = tup[2]*prob_sec
        smo_neu.append((tup[0], smo))
    smo_sub = sorted(smo_sub, key = lambda x: x[1], reverse=True)
    smo_neu = sorted(smo_neu, key = lambda x: x[1], reverse=True)
    return smo_sub,smo_neu

def select_words(smo_sub, smo_neu, number):
    words_sub = list()
    words_neu = list()
    for i in range(number):
        words_sub.append(smo_sub[i][0])
        words_neu.append(smo_neu[i][0])
    words = [x for x in words_sub if x not in words_neu]
    for word in words_neu:
        if word not in words_sub:
            words.append(word)
    return words

def generateSMO_data(X,Y,x_sub, x_neu):
    x_sub, total_sub = word_freq(x_sub)
    x_neu, total_neu = word_freq(x_neu)
    smo_sub, smo_neu  = calculate_cmfs(x_sub,x_neu, total_neu + total_sub)
    fit_words = select_words(smo_sub, smo_neu, number = 60)
    #calculate_smo(x_sub,x_neu, total_neu+total_sub)
    vectorizer = CountVectorizer()
    vectorizer.fit(fit_words)
    X = vectorizer.transform(X)
    my_path = os.path.split(os.path.abspath(__file__))[0]
    #save X and Y
    scipy.sparse.save_npz(os.path.join(my_path,'X.npz'), X)
    Y = np.array(Y)
    np.save(os.path.join(my_path,'Y'), Y)
    np.savetxt(os.path.join(my_path,'fit_words.txt'), fit_words, delimiter="  ", fmt="%s")
    #sparse_matrix = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')
