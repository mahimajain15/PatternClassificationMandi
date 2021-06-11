import csv
import random
import math
import operator
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from pandas import *
import numpy as np
from sklearn.metrics import classification_report 
import time
start_time = time.time()

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

def mahalanobisDistance(instance1, instance2, length , vari):
	distance = 0
	for x in range(length):
		distance += pow((float(instance1[x]) - float(instance2[x]))*vari[x][x], 2)
	return math.sqrt(distance)
 
def getNeighbors(trainingSet, testInstance, vari):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = mahalanobisDistance(testInstance, trainingSet[x], length, vari)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(len(distances)):
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
	#print(response)
	return sortedVotes[0][0]

 
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():

	#l00 = 0
	#l01 = 0
	#l02 = 0
	#l10 = 0
	#l11 = 0
	#l12 = 0
	#l20 = 0
	#l21 = 0
	#l22 = 0
	split = 0.70
	count10 = 0.0
	count11 = 0.0
	count20 = 0.0
	count21 = 0.0
	count30 = 0.0
	count31 = 0.0
	one = 0
	two = 0
	three = 0	

	cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
	# prepare data
	trainingSet=[]
	testSet=[]
	data=[]
	mean=[]
	split = 0.70
	# generate predictions
	predictions=[]
	y_test=[]
	y_pred=[]
	
	var1 = np.zeros((2,2), dtype=float)
	var2 = np.zeros((2,2), dtype=float)
	var3 = np.zeros((2,2), dtype=float)
	var_atr = np.zeros((2,2), dtype=float)
	var_gen = np.zeros((2,2), dtype=float)
	var_obs = np.zeros((2,2), dtype=float)

	handleDataset('/home/mahima/knn_ls_testtrain.txt', split, trainingSet, testSet)
	print ('Train: ' + repr(len(trainingSet)))
	print ('Test: ' + repr(len(testSet)))
	print ('Split in: ' + repr(split))
	
	for i in range(len(trainingSet)):
		if trainingSet[i][-1]== '0':
			count10 +=float(trainingSet[i][-3])
			count11 +=float(trainingSet[i][-2])
			one += 1
		if trainingSet[i][-1]== '1':
			count20 +=float(trainingSet[i][-3])
			count21 +=float(trainingSet[i][-2])
			two += 1
		if trainingSet[i][-1]== '2':
			count30 +=float(trainingSet[i][-3])
			count31 +=float(trainingSet[i][-2])
			three += 1
	mean10 = float(count10/one)
	mean11 = float(count11/one)
	mean20 = float(count20/two)
	mean21 = float(count21/two)
	mean30 = float(count30/three)
	mean31 = float(count31/three)
	#print('mean10: ' + str(mean10))

	with open('/home/mahima/mean_var_ls_data.txt', 'w') as csvfile:
		csvfile.write(str(mean10))
		csvfile.write(',')
		csvfile.write(str(mean11))
		csvfile.write(',')
		csvfile.write('0')
		csvfile.write('\n')
		csvfile.write(str(mean20))
		csvfile.write(',')
		csvfile.write(str(mean21))
		csvfile.write(',')
		csvfile.write('1')
		csvfile.write('\n')
		csvfile.write(str(mean30))
		csvfile.write(',')
		csvfile.write(str(mean31))
		csvfile.write(',')
		csvfile.write('2')
	
	with open('/home/mahima/mean_var_ls_data.txt', 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(len(dataset)):
			for y in range(3):
				dataset[x][y] = float(dataset[x][y])
	        #if random.random() < split:
			mean.append(dataset[x])
	       # else:
	          #  testSet.append(dataset[x])
	#print(mean)
	for col in range(0,2):
		for row in range(0,len(trainingSet)):
			if trainingSet[row][-1] == '0':
				var1[col][col] += pow((float(trainingSet[row][col]) - float(mean[0][col])), 2)
				#print(var1[col][col])
			if trainingSet[row][-1] == '1':
				var2[col][col] += pow((float(trainingSet[row][col]) - float(mean[1][col])), 2)
			if trainingSet[row][-1] == '2':
				var3[col][col] += pow((float(trainingSet[row][col]) - float(mean[2][col])), 2)

	#print('var1:')
	#print(var1)
	#print('var2:')
	#print(var2)
	#print('var3:')
	#print(var3)
	myInt1 =1/ 299
	var_atr = np.multiply(var1, myInt1)
	myInt2 =1/ 499	
	var_gen = np.multiply(var2, myInt2)
	myInt3 =1/ 799	
	var_obs = np.multiply(var3, myInt3)
	#print('var1:')
	#print(var_atr)
	#print('var2:')
	#print(var_gen)
	#print('var3:')
	#print(var_obs)	
	#print("var1")	
	#print(var_atr)
	var_atr = np.linalg.inv(var_atr) 
	var_gen = np.linalg.inv(var_gen) 
	var_obs = np.linalg.inv(var_obs) 
	#print("varinverse")	
	#print(var_atr)

	#print('Train set: ' + repr(len(trainingSet)))
	#print('Test set: ' + repr(len(testSet)))

	for x in range(len(testSet)):
		if testSet[x][-1] == '0':
			var = var_atr
		elif testSet[x][-1] == '1':
			var = var_gen
		elif testSet[x][-1] == '2':
			var = var_obs
		neighbors = getNeighbors(trainingSet, testSet[x], var)
		#print(neighbors)	
		result = getResponse(neighbors)
		#print(result)
		predictions.append(result)
		#print(predictions)
		if testSet[x][-1] =='0':
			if result =='0':
				cm[0][0] += 1
				#print(cm[0][0])
			elif result =='1':
				cm[0][1] += 1
			elif result == '2':
				cm[0][2] += 1
		if testSet[x][-1] =='1':
			if result =='0':
				cm[1][0] += 1
			elif result =='1':
				cm[1][1] += 1
			elif result =='2':
				cm[1][2] += 1
		if testSet[x][-1] =='2':
			if result =='0':
				cm[2][0] += 1
			elif result =='1':
				cm[2][1] += 1
			elif result =='2':
				cm[2][2] += 1
		y_test.append(testSet[x][-1])
		y_pred.append(result)
		#print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	print('Confusion Matrix:')
	print(DataFrame(cm, columns=['a', 'b', 'c'], index=['a', 'b', 'c']))
	print(classification_report(y_test, y_pred))
	
	
main()
print("Time Taken :: --- %s seconds ---" % (time.time() - start_time))  
