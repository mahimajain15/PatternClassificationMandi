import csv
import random
import math
import operator
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from pandas import *

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((float(instance1[x]) - float(instance2[x])), 2)
	return math.sqrt(distance)
 
def getNeighbors(trainingSet, testInstance, k):
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
	n = len(neighbors)
	for x in range(n):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
 
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
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

	# prepare data
	trainingSet=[]
	testSet=[]
	#split = 0.67
	with open('/home/mahima/image/knn_im_train.txt', 'r') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)):
	        #for y in range(4):
	         #   dataset[x][y] = float(dataset[x][y])
	        trainingSet.append(dataset[x])
	with open('/home/mahima/image/im_test.txt', 'r') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        testSet.append(dataset[x])
	print('Train set: ' + repr(len(trainingSet)))
	print('Test set: ' + repr(len(testSet)))
	# generate predictions
	predictions=[]
	k = input("Enter value of k: ")
	k = int(k) 
	print(k) 
	list_pred = []
	list_act = []
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		list_pred.append(result)
		list_act.append(testSet[x][-1])
	#print('predicted list: ' + repr(list_pred) + repr(len(list_pred))) 
	#print('actual list: ' + repr(list_act) + repr(len(list_act)))
		#print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	#print(list_pred)	
	for i in range(0, 147):
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
	print(DataFrame(a, columns = ['botanical_garden', 'bus_interior', 'elevater_shaft'], index = ['botanical_garden', 'bus_interior', 'elevater_shaft']))
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

	num_a = [1, 3, 5, 7, 9, 11, 13, 15]
	num_b = [42.18, 44.22, 49.66, 50.34, 52.38, 51.70, 49.66, 50.34]
	plt.plot(num_a, num_b)
	plt.show()
main()
