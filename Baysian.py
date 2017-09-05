'''
@author: msi
'''
import numpy as np
import random
import matplotlib.pyplot as plot
import csv
import copy
import math
from asyncore import write


#data access
def loadCsv(filename):
    lines=csv.reader(open(filename,"rb"))
    dataset=list(lines)
    for i in range(len(dataset)):
        dataset[i]=[float(x) for x in dataset[i]]
    return dataset

#write data into disk
def writeCsv(filename,dataset,columns):
    csvfile=file(filename,'wb+')
    writer=csv.writer(csvfile)
    data=dataset
    if columns==2:
        writer.writerow(['X','Results'])
        writer.writerows(data)
    if columns==1:
        writer.writerow(['Predictions'])
        writer.writerow(data)
  
    csvfile.close()

#spilt dataset into trainset and testset
def splitDataset(dataset,splitRatio):
    trainSize = int(len(dataset)*splitRatio)
    trainSet=[]
    copy=list(dataset)
    while len(trainSet)<trainSize:
        index=random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet,copy]


def separateByClass(dataset):
    separate={}
    for i in range(len(dataset)):
        vector=dataset[i]
        if(vector[-1] not in separate):
            separate[vector[-1]]=[]
        separate[vector[-1]].append(vector)
    return separate

#calculate mean and variance
def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg=mean(numbers)
    variance=sum([pow(x-avg, 2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

#get summarise of dataset
def summarise(dataset):
    summarise=[(mean(attribute),stdev(attribute)) for attribute in zip(*dataset)]
    del summarise[-1]
    return summarise

#calculate piror probability
def calculatePirorProbability(dataset):
    count=0
    for i in range(len(dataset)):
        vector=dataset[i]
        if (int(vector[-1]) == 0):
            count+=1
    
    return  ((count)/float(len(dataset)))
       
#calaulate Gauss probability-density function
def calculateProbability(x,mean,stdev):
    exponent=math.exp(-(math.pow(x-mean, 2)/(2*(math.pow(stdev, 2)))))
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent

#calculate class  probabilities function
def calculateClassProbabilities(summaries,inputVector):
    probabilities={}
    for classValue,classSummaries in summaries.iteritems():
        probabilities[classValue]=calculatePirorProbability(dataset)
        for i in range(len(classSummaries)):
            mean,stdev=classSummaries[i]
            x=inputVector[i]
            probabilities[classValue]=(probabilities[classValue]*calculateProbability(x, mean, stdev))/(1-probabilities[classValue])
    return probabilities

#predict biggest probabilities
def predict(summaries,inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)    
    bestMatch,bestPro = None,-1
    for classValue,probability in probabilities.iteritems():
        if bestMatch is None or probability> bestPro:
            bestPro=probability
            bestMatch=classValue
    return bestMatch

#predict testsets
def getPredictions(summaries,testSet):
    predictions=[]
    for i in range(len(testSet)):
        result=predict(summaries, testSet[i])
        predictions.append(result)
    return predictions
    

#get accurancy
def getAccurancy(testSet,predictions):
    correct=0
    for i in range(len(testSet)):
        if testSet[i][-1]==predictions[i]:
            correct+=1
    return (correct/float(len(testSet)))*100.00

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarise(instances)
    return summaries
'''
def combineDataset(dataset1,dataset2):  
    for i in range(len(dataset1)):
        combineSet=dataset1[i]
        print type(dataset[i])
    return combineSet  
 '''
filename='2016.csv'
splitRatio=0.67
dataset=loadCsv(filename)
trainingSet,testSet=splitDataset(dataset, splitRatio)
print ('Split {0} rows into  {1} training rows and test with {2} rows').format(len(dataset),len(trainingSet),len(testSet))
summaries=summarizeByClass(trainingSet)


#print summaries    
predictions=getPredictions(summaries,testSet)
accuracny=getAccurancy(testSet,predictions)


print('accurancy: {0}').format(accuracny)
print predictions
#combine sets
#resultSet=combineDataset(testSet, predictions)


#wirte data
#writeCsv('2016trainingSet(1).csv', trainingSet,2)
#writeCsv('2016testSet(1).csv', testSet,2)
#writeCsv('2016resultSet(1).csv',predictions,1)
