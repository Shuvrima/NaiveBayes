#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 21:32:40 2019

@author: shuvrimaalam
"""

import pandas
import operator


dataframe_train = pandas.DataFrame()          # declare empty dataframe


def read_file():
    dataframe_train = pandas.read_csv("Text_attribues_data_for_training.txt", delimiter=r"\s+")
    dataframe_test = pandas.read_csv("testfile.txt", delimiter=r"\s+")
    test_list = []
    train_list = []
    l1 = []

    for y in dataframe_train.itertuples():
        l1 = [i for i in y[1:]]
        train_list.append(l1)
    for z in dataframe_test.itertuples():
                test_list = [j for j in z[1:]]
    vocab = []
    for col in dataframe_train.columns:
        vocab.append(len(dataframe_train[col].unique()))  
    print(vocab)

    return train_list, test_list, dataframe_train,dataframe_test,vocab


def prior(train_list):
    label = {}               
    prior_prob = {}
    N = len(train_list)  
    p = 0
    for item in train_list:
      
        if item[-1] not in label:   
          
            label[item[-1]] = 1      # found item for first time so 1
            #print('label', label)
        else:
            #print('item if there ', item[-1])
            label[item[-1]] += 1      # inc class freq value
            #print('label if there', label)

    print(label)
    for i in label:
        Nc = int(label[i])    #mammal:3 this will take 3  label[mammal]=3
        p = round(float(Nc/N), 2)      # cal 3/8
        prior_prob[i] = p           # for same label mammal=p append to new dict priorprob\\

    #print(prior_prob)
    return prior_prob,label

def cond(c,d,v,label):
    d1 = {}
    len = [1, 2, 3, 4, 5, 6]    # the 0th col is id
    for l in label:         # l iteratinng thru the label dict with the class freq
        i = 0
        prod1 = 1
        prod2 = 1
        print(l)
        for col, n in zip(c.columns, len):        # col taking each col name from training dataframe and n iterating value from len & running simultaneously
            print(col, n)
            df = (c.groupby('Class')[col].value_counts())     # df of c
            df = df.to_frame()
            for g in d.itertuples():         # g is the list of test data g=[gila-monster, cold-blooded ,scales no yes yes] d is test dataframe
                print('hello',g[n])        # g[1]=gila as g[0] will be id
                for f in df.itertuples():  # df is train dataframe and f = each list in the dataframe so f[][] will be the value and f[1] will be the count
                    print(f)
                    if (f[0][1] == g[n] and l == f[0][0]):   # Pandas(Index=('amphibian', 'no'), GivesBirth=1) g(n) is taking value from testdata from att to att till end
                        print('f number', f[0][1], f[0][0], f[1], l)
                        niy = f[1]
                        print('niy', niy, label[l], v[i])
                        lap = round(float((niy + 1) / (label[l] + v[i])), 4)  #
                        prod1 *= lap
                        print('prod1', prod1)
            i += 1
        d1[l] = float(prod1 * prod2)
    print(d1)
    return d1


def posterior(prior_prob,d1):
    print(prior_prob)
    d2 = {}
    x=[]
    for i,j in zip(prior_prob,d1):
        post=float(prior_prob[i]* d1[j])
        d2[j] = post
        print(d2)
    import operator   
    print('Maximum:',max(d2.items(), key=operator.itemgetter(1))[0])


def findProbability(data):
    countClass = dict()
    for value in data:
      if value in countClass:
        countClass[value] += 1
      else:
        countClass[value] = 1

    N = len(data)
    probability = dict()
    for key in countClass:
      probability[key] = countClass[key]/N

    return probability
    conditional = dict( )
    for className in Y_prior:
      conditional[className] = dict()
      classDataIndex = np.where(Y_train == className)[0]
      i = 0
      for attribute in range( M ):
        conditional[ className ][ headers[attribute] ] = findProbability( X_train[classDataIndex, attribute] )

if __name__ == "__main__":
    dt1,dt2,c,d,v = read_file()
    p1,p2 = prior(dt1)
    co=cond(c,d,v,p2)
    posterior(p1,co)