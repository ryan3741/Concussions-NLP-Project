import random
import collections
import math
import sys
import csv
import os, random, operator, sys
from collections import Counter


def getBest(weights, wordDict):
    string = ''
    bestWeight = float('-inf')
    uses = 0
    for word in weights:
        if (wordDict[word] > 10):
            if weights[word] > bestWeight:
                string = word
                bestWeight = weights[word]
                uses = wordDict[word]
    return((string, bestWeight, uses))

def getWorst(weights, wordDict):
    string = ''
    bestWeight = float('inf')
    uses = 0
    for word in weights:
        if (wordDict[word] > 10):
            if weights[word] < bestWeight:
                string = word
                bestWeight = weights[word]
                uses = wordDict[word]
    return((string, bestWeight, uses))

def getFive(weights, wordDict):
    topFive = []
    bottomFive = []
    modifiedWeights = weights
    for i in range(10):
        string, weight, uses = getBest(modifiedWeights, wordDict)
        if string != '':
            del modifiedWeights[string]
        topFive.append((string, uses, weight))

        string, weight, uses = getWorst(modifiedWeights, wordDict)
        if string != '':
            del modifiedWeights[string]
        bottomFive.append((string, uses, weight))
    return((topFive, bottomFive))

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
    wordsToRemove = [
        'concussion', 'laceration', 'contusion', 'lac', ' co ', 'chi', 'conc','closed','clsd','sdh','cont','sah','subdural',"h'tma'","iph",'ich','usison','ussive',
        '*', 'yr', '-', ';', ',', '.', '>', '<', ':', '/', '+', '"',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '?', '#'
        ]
    for word in wordsToRemove:
        newstring = newstring.replace(word, ' ')

    # Replaces several words    
    replaceDict = {
        'pt':' patient ', ' loss of conciousness ':' loc ', 'yom':' male ', 'yof':' female ', ' m ':' male ', 
        ' f ':' female ', ' l ':' left ', 'lf':' left ', ' r ':' right ', ' rt ':' right ', ' inj ': ' injury ',
        'h/a':' headache ', 'w/o':' without ', 'w/':' with ', '@':' at ', '&':' and '
        }
    for key in replaceDict:
        newstring = newstring.replace(key, replaceDict[key])
    return newstring

def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in list(d2.items()))


def extractWordFeatures(narrative, ageFeature, raceFeature, sexFeature):
    """
    Extract word features for a string narractive. Words are delimited by
    whitespace characters only.
    @param string x:
    @return wordDict: feature vector representation of x and ageFeature.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    wordList = narrative.split()
    wordDict = collections.defaultdict(lambda:0)
    counter = 0
    for word in wordList:
        counter += 1
        # Extracts one word features
        wordDict[word] += 1

        # Extracts two word features
        # if counter < len(wordList) - 1:
          #  nextWord = wordList[counter + 1]
          #  string = word + ' ' + nextWord
          #  wordDict[string] += 1
    
    if ageFeature == '':
        age = 0
    else: 
        age = int(ageFeature)
    ageCategories = [(2,5),(6,10),(11,15),(16,20),(21,30),
        (31,40),(41,50),(51,60),(61,70),(71,80),(81,90),(91,120),(200,224)]
    for ageRange in ageCategories:
        minAge, maxAge = ageRange
        if age >= minAge and age <= maxAge:
            if age >= 200:
                wordDict["0-1"] += 1
            else: 
                wordDict["{}-{}".format(minAge,maxAge)] += 1 
    
    if raceFeature == '':
        race = 0
    else:
        race = int(raceFeature)
    raceCategories = [(0, "no race"),(1, "white"), (2, "black"), (3, "other"),
        (4, "asian"), (5, "american indian"), (6, "native hawaiian")]
    for elem in raceCategories:
        raceId, raceTitle = elem
        if race == raceId:
            wordDict[raceTitle] += 1

    if sexFeature == '':
        sex = 0
    else:
        sex = int(sexFeature)
    sexCategories = [(0, "unknown sex"), (1, "male sex"), (2, "female sex")]
    for elem in sexCategories:
        sexId, sexTitle = elem
        if sex == sexId:
            wordDict[sexTitle] += 1

    return wordDict


def evaluatePredictor(examples, predictor):
    '''
    predictor: a function that takes an x and returns a predicted y.
    Given a list of examples (x, y), makes predictions based on |predict| and returns the fraction
    of misclassiied examples.
    '''
    error = 0
    falsePos = 0
    falseNeg = 0
    truePos = 0
    trueNeg = 0

    for narrative, age, race, sex, y in examples:
        xValue = predictor(narrative, age, race, sex)
        if xValue != y:
            error += 1
        if xValue == 1 and y ==1:
            truePos += 1
        if xValue == 1 and y == -1:
            falsePos += 1
        if xValue == -1 and y == -1:
            trueNeg += 1
        if xValue == -1 and y ==1:
            falseNeg += 1
    sens = truePos/(truePos+falseNeg)
    spec = trueNeg/(trueNeg+falsePos)
    print("Hinge Loss")
    print("Sensitivity: ", sens)
    print("Specificity: ", spec)
    print("True Positive: ", truePos)
    print("True Negative: ", trueNeg)
    print("False Positive: ", falsePos)
    print("False Negative: ", falseNeg)
    return 1.0 * error / len(examples)

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x, race, age, sex,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    for t in range(numIters):
        for currTrain in trainExamples:
            features = featureExtractor(currTrain[0], currTrain[1], currTrain[2], currTrain[3])
            margin = dotProduct(weights, features)
            hinge = margin*currTrain[4]
            if hinge < 1:
                for feature in features:
                    if feature in weights:
                        weights[feature] += eta*features[feature]*currTrain[4]
                    else:
                        weights[feature] = eta*features[feature]*currTrain[4]
    print("Train error is:")
    print(evaluatePredictor(trainExamples, lambda w, x, y, z : (1 if dotProduct(featureExtractor(w, x, y, z), weights) >= 0 else -1)))
    print("")
    print("Test error is:")
    print(evaluatePredictor(testExamples, lambda w, x, y, z : (1 if dotProduct(featureExtractor(w, x, y, z), weights) >= 0 else -1)))
    print("")
    return weights

#trainExamples = [("yes and",1),("no",-1), ("yes and",1), ("nah",-1)]
#testExamples = [("yes and",1),("no",-1), ("yes and",-1)]



def readData(pathArray):   
    labeledData = []
    wordDict = collections.defaultdict(lambda:0)
    for path in pathArray:
        input_file = csv.DictReader(open(path))
        for row in input_file:
            string = row['Narrative_1'] + row['Narrative_2']
            newString = clean(string)
            age = row['Age']
            race = row['Race']
            sex = row['Sex']
            if row['Diagnosis'] == '52':
                labeledData.append((newString, age, race, sex, 1))
            else:
                labeledData.append((newString, age, race, sex, -1))

            wordList = newString.split()
            for word in wordList:
                wordDict[word] += 1
    return labeledData, wordDict


trainDataPath = ["/Users/ryancrowley/Desktop/Project221/NEISS_2017.csv"]
testDataPath = ["/Users/ryancrowley/Desktop/Project221/NEISS_2018.csv"]

#trainDataPath = [
#"/Users/ryancrowley/Desktop/Project221/NEISS_2009.csv", 
#"/Users/ryancrowley/Desktop/Project221/NEISS_2010.csv",
#"/Users/ryancrowley/Desktop/Project221/NEISS_2011.csv", 
#"/Users/ryancrowley/Desktop/Project221/NEISS_2012.csv",
#"/Users/ryancrowley/Desktop/Project221/NEISS_2013.csv",
#"/Users/ryancrowley/Desktop/Project221/NEISS_2014.csv", 
#"/Users/ryancrowley/Desktop/Project221/NEISS_2015.csv",
#"/Users/ryancrowley/Desktop/Project221/NEISS_2016.csv"]

#testDataPath = [
#"/Users/ryancrowley/Desktop/Project221/NEISS_2017.csv",
#"/Users/ryancrowley/Desktop/Project221/NEISS_2018.csv"]


trainData, trainWordCount = readData(trainDataPath)
testData, testWordCount = readData(testDataPath)

weights = learnPredictor(trainData,testData,extractWordFeatures,100,.01)
topFive, bottomFive = getFive(weights, trainWordCount)
print("------")
for elem in topFive:
    print(elem)
print("------")
for elem in bottomFive:
    print(elem)

#print(weights)