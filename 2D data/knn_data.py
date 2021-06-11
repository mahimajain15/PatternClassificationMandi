import math
import random
import csv
import numpy as np
from pandas import *

trainingSet=[]
testSet=[]
with open(r'/home/mahima/2D data/knn_data_testtrain.txt') as csvfile:
    lines = csv.reader(csvfile)
    #for row in lines:
        #print (', '.join(row))

def handleDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
#            for y in range(4):
#                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


#handleDataset('/home/mahima/group03.txt', 0.66, trainingSet, testSet)
#print ('Train: ' + repr(len(trainingSet)))
#print ('Test: ' + repr(len(testSet)))

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((float(instance1[x]) - float(instance2[x])), 2)
    return math.sqrt(distance)

import operator 
def getKNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    #print(sortedVotes[0][0])
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] is predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

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
	split = 0.70
	handleDataset('/home/mahima/2D data/knn_data_testtrain.txt', split, trainingSet, testSet)
	print(trainingSet)
	print ('Train: ' + repr(len(trainingSet)))
	print ('Test: ' + repr(len(testSet)))
	print ('Split in: ' + repr(split))
	#generate predictions
	predictions = []
	k = input("Enter value of k: ")
	k = int(k) 
	list_pred = []
	list_act = []
	for x in range(len(testSet)):
		neighbors = getKNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		list_pred.append(result)
		list_act.append(testSet[x][-1])
		#print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%') 
	for i in range(0, len(testSet)):
		x = list_pred[0]
		y = list_act[0]
		if int(list_pred[i]) == 0 and int(list_act[i]) == 0:
			l00 += 1
		elif int(list_pred[i]) == 0 and int(list_act[i]) == 1:
			l01 += 1
		elif int(list_pred[i]) == 0 and int(list_act[i]) == 2:
			l02 += 1
		elif int(list_pred[i]) == 1 and int(list_act[i]) == 0:
			l10 += 1
		elif int(list_pred[i]) == 1 and int(list_act[i]) == 1:
			l11 += 1
		elif int(list_pred[i]) == 1 and int(list_act[i]) == 2:
			l12 += 1
		elif int(list_pred[i]) == 2 and int(list_act[i]) == 0:
			l20 += 1
		elif int(list_pred[i]) == 2 and int(list_act[i]) == 1:
			l21 += 1
		elif int(list_pred[i]) == 2 and int(list_act[i]) == 2:
			l22 += 1
		i+=1
	a = np.array([[l00, l01, l02],
		[l10, l11, l12],
		[l20, l21, l22]])
	print('Confusion Matrix: ')	
	print(DataFrame(a, columns = ['class_0', 'class_1', 'class_2'], index = ['class_0', 'class_1', 'class_2']))
	#print(a)
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
