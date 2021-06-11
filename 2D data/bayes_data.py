import csv
import math
import random
import numpy as np 
from pandas import *

trainSet = []
trainSet1 = []
trainSet2 = []
trainSet3 = []
def loadCsv(filename):
	lines = csv.reader(open(r'/home/mahima/2D data/knn_data_testtrain.txt'))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	prob =(1/(math.sqrt(2*math.pi)*stdev))*exponent
	return prob

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
		
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet)))*100.0

def main():
	l00 = 0
	l01 = 0
	l02 = 0
	l10 = 0
	l11 = 0
	l12 = 0
	l20 = 0
	l21 = 0
	l22 = 0
	list_pred = []
	list_act = []
	filename = 'knn_data_testtrain.txt'
	splitRatio = 0.67
	dataset = loadCsv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	print('Split {0} rows into train = {1} and test = {2} rows'.format(len(dataset),len(trainingSet),len(testSet)))
	#print(np.shape(np.cov(np.transpose(trainingSet))))
		
	for x in range(len(testSet)):
		#prepare model
		summaries = summarizeByClass(trainingSet)
		#test model
		predictions = getPredictions(summaries, testSet)		
		list_act.append(testSet[x][-1])
	accuracy = getAccuracy(testSet, predictions)
	#print(predictions)
	print('Accuracy: {0}%'.format(accuracy))
	#print(testSet)
	#print(len(list_act))
	for i in range(0, len(testSet)):		
		x = predictions[0]
		y = list_act[0]
		#print(list_act)
		if predictions[i] == 0 and list_act[i] == 0:
			l00 += 1
		elif predictions[i] == 0 and list_act[i] == 1:
			l01 += 1
		elif predictions[i] == 0 and list_act[i] == 2:
			l02 += 1
		elif predictions[i] == 1 and list_act[i] == 0:
			l10 += 1
		elif predictions[i] == 1 and list_act[i] == 1:
			l11 += 1
		elif predictions[i] == 1 and list_act[i] == 2:
			l12 += 1
		elif predictions[i] == 2 and list_act[i] == 0:
			l20 += 1
		elif predictions[i] == 2 and list_act[i] == 1:
			l21 += 1
		elif predictions[i] == 2 and list_act[i] == 2:
			l22 += 1
		i+=1
	a = np.array([[l00, l01, l02],
		[l10, l11, l12],
		[l20, l21, l22]])
	print('Confusion Matrix: ')	
	print(DataFrame(a, columns = ['class_0', 'class_1', 'class_2'], index = ['class_0', 'class_1', 'class_2']))
	prec_0 = (l00/float(l00+l10+l20))
	prec_1 = (l11/float(l01+l11+l21))
	prec_2 = (l22/float(l02+l12+l22))
	print('Precision: ')
	print('Precision for class 0: ' + repr(prec_0))
	print('Precision for class 1: ' + repr(prec_1))
	print('Precision for class 2: ' + repr(prec_2))
	print('Average Precision: ' + repr((prec_0+prec_1+prec_2)/3))

	rec_0 = (l00/float(l00+l01+l02))
	rec_1 = (l11/float(l10+l11+l12))
	rec_2 = (l22/float(l20+l21+l22))
	print('Recall: ')
	print('Recall for class 0: ' + repr(rec_0))
	print('Recall for class 1: ' + repr(rec_1))
	print('Recall for class 2: ' + repr(rec_2))
	print('Average Recall: ' + repr((rec_0+rec_1+rec_2)/3))

	f0 = (2*(prec_0*rec_0)/(prec_0+rec_0))
	f1 = (2*(prec_1*rec_1)/(prec_1+rec_1))
	f2 = (2*(prec_2*rec_2)/(prec_2+rec_2))
	print('F1 Score: ')
	print('F1 Score for class 0: ' + repr(f0))
	print('F1 Score for class 1: ' + repr(f1))
	print('F1 Score for class 2: ' + repr(f2))
	print('Average F1 Score: ' + repr((f0+f1+f2)/3))
main()
