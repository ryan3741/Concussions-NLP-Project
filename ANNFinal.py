import numpy as np
import random
import collections
import math
import sys
import numpy as np
import csv
import os, random, operator, sys
from collections import Counter


def clean(string):
    newstring = string.lower()

    # Removes anything after 'dx' or 'd:'
    index = newstring.find('dx')
    if index != -1:
        newstring = newstring[:index]
        index = newstring.find('d:')
    if index != -1:
        newstring = newstring[:index]

    # Removes 'concussion', 'laceration', 'contusion' from string
    #wordsToRemove = ['concussion', 'laceration', 'contusion', 'lacerated', '*', 'lac', 'chi']
    #wordsToRemove = []
    #newString 
    #for word in wordsToRemove:
        #newstring = newstring.replace(word, ' ')

    # Replaces several words
    newstring = newstring.replace('pt', ' patient ')
    newstring = newstring.replace(' loss of conciousness ', ' loc ')
    #newstring = newstring.replace('loc', ' loss of conciousness ')
    newstring = newstring.replace('yom', ' male ')
    newstring = newstring.replace('yof', ' female ')
    newstring = newstring.replace(' m ', ' male ')
    newstring = newstring.replace(' f ', ' female ')
    newstring = newstring.replace('yr', '')
    newstring = newstring.replace(' l ', ' left ')
    newstring = newstring.replace('lf', ' left ')
    newstring = newstring.replace(' r ', ' right ')
    newstring = newstring.replace(' rt ', ' right ')
    newstring = newstring.replace(' inj ', ' injury ')
    newstring = newstring.replace('h/a', ' headache ')
    newstring = newstring.replace('w/o', ' without ')
    newstring = newstring.replace('w/', ' with ')
    newstring = newstring.replace('@', ' at ')
    newstring = newstring.replace('&', ' and ')
    newstring = newstring.replace('-', ' ')
    newstring = newstring.replace(';', ' ')
    newstring = newstring.replace(',', ' ')
    newstring = newstring.replace('.', ' ')
    newstring = newstring.replace('>', ' ')
    newstring = newstring.replace(':', ' ')
    newstring = newstring.replace('/', ' ')
    newstring = newstring.replace('+', ' ')
    newstring = newstring.replace('"', ' ')
    newstring = newstring.replace(' co ', ' ')
    return newstring

yLabel = []
input_file = csv.DictReader(open("/Users/ryancrowley/Desktop/Project221/evensplit.csv"))
trainData = []
wordDict = collections.defaultdict(lambda:0)
counter = 0
for row in input_file:
    string = row['Narrative_1'] + row['Narrative_2']
    newString = clean(string)
    if row['Diagnosis'] == '52':
        yLabel.append(1)
    else:
        yLabel.append(0)

    wordList = newString.split()
    for word in wordList:
        if word not in wordDict:
            wordDict[word] = counter
            counter += 1

input_file = csv.DictReader(open("/Users/ryancrowley/Desktop/Project221/trialData.csv"))
for row in input_file:
    string = row['Narrative_1'] + row['Narrative_2']
    newString = clean(string)
    wordList = newString.split()
    currArray = [0] * len(wordDict)
    for word in wordList:
        currArray[wordDict[word]] +=1
    trainData.append(currArray)


np.random.seed(3)
#feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
feature_set = np.array(trainData)
#labels = np.array([[1,0,0,1,1]])
labels = np.array([yLabel])
labels = labels.reshape(len(yLabel),1)

whidden = np.random.rand(len(feature_set[0]),4) 
wout = np.random.rand(4, 1)
#lr = .1
lr = .01

#lr = .01 and 4000 currEpochs works for evensplit and trialData
#lr = .01 and 20000 currEpochs works for trialData and evensplit


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))


for currEpoch in range(20000):
    # feedforward
    zhidden = np.dot(feature_set, whidden)
    ahidden = sigmoid(zhidden)

    zout= np.dot(ahidden, wout)
    aout = sigmoid(zout)

  
    currError = ((1 / 2) * (np.power((aout - labels), 2)))
    print(currError.sum())

    costOut = aout - labels
    ZDeriv = sigmoid_der(zout) 
    currDeriv = ahidden

    costWout = np.dot(currDeriv.T, CostAOut * zoutDeriv)
  

    # costWout= ZDeriv * DerivAhidden
    # costDHidden = costDZ * ZoutAHidden

    costDZ = CostAOut * zoutDeriv
    ZoutAHidden = wout
    costDHidden = np.dot(costDZ , ZoutAHidden.T)
    DerivZhidden = sigmoid_der(zhidden) 
    currFeatures = feature_set
    dcost_wh = np.dot(currFeatures.T, DerivZhidden * costDHidden)


    whidden -= lr * dcost_wh
    wout -= lr * costWout


incorrect = 0
yLabel = []
input_file = csv.DictReader(open("/Users/ryancrowley/Desktop/Project221/evensplit.csv"))
trainData = []

for row in input_file:
    string = row['Narrative_1'] + row['Narrative_2']
    newString = clean(string)
    if row['Diagnosis'] == '52':
        yLabel.append(1)
    else:
        yLabel.append(0)

    wordList = newString.split()
    currArray = [0] * len(wordDict)
    for word in wordList:
        if word in wordDict:
            currArray[wordDict[word]] +=1
    trainData.append(currArray)


for index in range(0,len(trainData)):
    single_point = np.array(trainData[index])

    zhidden = np.dot(single_point, whidden)
    ahidden = sigmoid(zhidden)
    zout= np.dot(ahidden, wout)
    aout = sigmoid(zout)
    print(aout)
    output = 0
    if aout > .5:
        output = 1
    if output != yLabel[index]:
        incorrect += 1
testError = incorrect/len(trainData) 
print(testError)

