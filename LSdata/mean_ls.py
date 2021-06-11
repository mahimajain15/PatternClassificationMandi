import csv
import random
import math
import operator
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from pandas import *
import pandas as pd
meanSet = []
def handleDataset(filename, split, trainingSet=[] , testSet=[]):
	csvfile = open(filename, 'r')
	lines = csv.reader(csvfile)
	dataset = list(lines)
	for x in range(len(dataset)):
		if random.random() < split: 
			trainingSet.append(dataset[x])
		else: 
			testSet.append(dataset[x]) 
	csvfile.close()
 
 
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((float(instance1[x]) - float(instance2[x])), 2)
	#print math.sqrt(distance)
	return math.sqrt(distance)
 
def getNeighbors(trainingSet, testInstance):
	distances = []
	length = len(testInstance)-1
	for x in range(len(meanSet)):
		dist = euclideanDistance(testInstance, meanSet[x], length)
		distances.append((meanSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(len(distances)):
		neighbors.append(distances[x][0])
	return neighbors
 
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		#print(response)
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
 
def getAccuracy(testSet, predicted):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predicted[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
# prepare data 
	cm = [[0,0,0],[0,0,0],[0,0,0]]
	trainingSet=[] 
	testSet=[] 
	split = 0.67 
	predicted = []
	actual = []
	count10 = 0.0
	count11 = 0.0
	count20 = 0.0
	count21 = 0.0
	count30 = 0.0
	count31 = 0.0
	one = 0
	two = 0
	three = 0
	with open("mean_ls_data.txt", "r") as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(len(dataset)):
			meanSet.append(dataset[x])
	handleDataset('/home/mahima/LSdata/knn_ls_testtrain.txt', split, trainingSet, testSet) 
	print('Train set: ' + repr(len(trainingSet))) 
	print('Test set: ' + repr(len(testSet))) 
	# generate predictions 
	predictions=[] 
	#print(trainingSet)
	#for i in range(len(trainingSet)):
	for i in range(len(trainingSet)):
		if trainingSet[i][-1] == '0':
			count10 += float(trainingSet[i][-3])
			count11 += float(trainingSet[i][-2])
			one += 1
		if trainingSet[i][-1] == '1':
			count20 += float(trainingSet[i][-3])
			count21 += float(trainingSet[i][-2])
			two += 1
		if trainingSet[i][-1] == '2':
			count30 += float(trainingSet[i][-3])
			count31 += float(trainingSet[i][-2])
			three += 1
	mean10 = float(count10/one)
	mean11 = float(count11/one)
	mean20 = float(count20/two)
	mean21 = float(count21/two)
	mean30 = float(count30/three)
	mean31 = float(count31/three)
	print("mean10 = " +str(mean10))
	print("mean11 = " +str(mean11))
	print("mean20 = " +str(mean20))
	print("mean21 = " +str(mean21))
	print("mean30 = " +str(mean30))
	print("mean31 = " +str(mean31))
	with open("mean_ls_data.txt","w") as mean:
		mean.write(str(mean10))
		mean.write(",")
		mean.write(str(mean11))
		mean.write(",0")
		mean.write("\n")
		mean.write(str(mean20))
		mean.write(",")
		mean.write(str(mean21))
		mean.write(",1")
		mean.write("\n")
		mean.write(str(mean30))
		mean.write(",")
		mean.write(str(mean31))
		mean.write(",2")
	#print(trainingSet[1])
	#print(testSet)
	for x in range(len(testSet)): 
		neighbors = getNeighbors(meanSet, testSet[x]) 
		result = getResponse(neighbors) 
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
		predicted.append(result)
		actual.append(testSet[x][-1])
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	for i in range(0,len(testSet)):
		if predictions[i] == '0' and actual[i] == '0':
			cm[0][0] += 1
		elif predictions[i] == '0' and actual[i]== '1':
			cm[0][1] += 1
		elif predictions[i] == '0' and actual[i]== '2':
			cm[0][2] += 1
		elif predictions[i] == '1' and actual[i]== '0':
			cm[1][0] += 1
		elif predictions[i] == '1' and actual[i]== '1':
			cm[1][1] += 1
		elif predictions[i] == '1' and actual[i]== '2':
			cm[1][2] += 1
		elif predictions[i] == '2' and actual[i]== '0':
			cm[2][0] += 1
		elif predictions[i] == '2' and actual[i]== '1':
			cm[2][1] += 1
		elif predictions[i] == '2' and actual[i]== '2':
			cm[2][2] += 1
	print(pd.DataFrame(cm, columns = ["cls1","cls2","cls3"], index =  ["cls1","cls2","cls3"]))
	r_0 = cm[0][0]/(float(cm[0][0]+cm[0][1]))
	r_1 = cm[1][1]/(float(cm[1][1]+cm[1][0]+cm[1][2]))
	r_2 = cm[2][2]/(float(cm[2][2]+cm[2][1]))
	print('R_0 = ' + str(r_0))
	print("R_1 = " + str(r_1))
	print("R_2 = " + str(r_2))
	recall = float(r_0 + r_1 + r_2)/3
	print("Recall = " + str(recall))
	p_0 = cm[0][0]/(float(cm[0][0]+cm[1][0]))
	p_1 = cm[1][1]/(float(cm[1][1]+cm[0][1]+cm[2][1]))
	p_2 = cm[2][2]/(float(cm[2][2]+cm[1][2]))
	precision = float(p_0 + p_1 + p_2)/3
	print("P_0 = " + str(p_0))
	print("P_1 = " + str(p_1))
	print("P_2 = " + str(p_2))
	print("Avg Precision : " + str(precision))
	f_0 = 2*(float(r_0*p_0))/float(r_0+p_0)
	f_1 = 2*(float(r_1*p_1))/float(r_1+p_1)
	f_2 = 2*(float(r_2*p_2))/float(r_2+p_2)
	f_avg = float(f_0+f_1+f_2)/3
	#f1_scr = 2*(float(recall*precision))/(float(recall+precision))
	print("f_0 = " + str(f_0))
	print("f_1 = " + str(f_1))
	print("f_2 = " + str(f_2))
	print("AVG F1_SCORE : " + str(f_avg))
	#print("F1_SCORE is : " + str(f1_scr))
	
	     

main()
