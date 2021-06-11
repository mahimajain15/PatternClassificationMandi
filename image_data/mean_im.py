"""import pandas as pd
import csv
import csv
import random
import math
import operator
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from pandas import *

url = "/home/mahima/t.txt"

# Assign colum names to the dataset
#names = ['x', 'y', 'class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url)  
#dataset.head()
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 24].values
X_train = X
y_train = y
print X.shape
print y.shape
#now for test data
url = "/home/mahima/test.txt"

# Assign colum names to the dataset
#names = ['x', 'y', 'class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url)  
#dataset.head()
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 24].values
X_test = X
y_test = y
print X.shape
print y.shape
#using sklearn
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=1)  
classifier.fit(X_train, y_train)  
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))
scores = []
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
scores.append(metrics.accuracy_score(y_test, y_pred))

print(scores)


# Calculating error for K values between 1 and 40


"""

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
 
def getNeighbors(trainingSet, testInstance):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(len(distances)):
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
	with open('/home/mahima/image/mean_im_t.txt', 'r') as csvfile:
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
#	k = input("Enter value of k: ")
#	k = int(k) 
#	print(k) 
	list_pred = []
	list_act = []
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x])
		result = getResponse(neighbors)
		predictions.append(result)
		list_pred.append(result)
		list_act.append(testSet[x][-1])
	#print('predicted list: ' + repr(list_pred) + repr(len(list_pred))) 
	#print('actual list: ' + repr(list_act) + repr(len(list_act)))
		#print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	
	
	
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
main()




"""
import csv
import math
import operator
import numpy as np
from decimal import Decimal

count = 0.0
list_mean = []
j = 0
with open('/home/mahima/t.txt', 'r') as csvfile:
	lines = csvfile.read()
	t = lines.split(',')
	for i in range(25):
		for j in range(i, len(t), 25):
			#print(t[j])
			count += int(t[j])
			j +=1
		mean = float(count)/50.0
		list_mean.append(mean)
		count = 0.0
	print(list_mean)

[368907.82, 395409.84, 289508.8, 161285.02, 95095.12, 54168.62, 51320.0, 122881.66, 132888.5, 276319.3, 324929.84, 272794.74, 195509.18, 128639.22, 95026.84, 112469.26, 146322.92, 281219.02, 301103.18, 263315.1, 199861.74, 137878.52, 101388.78, 107487.62, 0.0]
[72180.4, 113147.18, 108597.9, 110749.36, 64599.72, 49433.82, 32040.62, 55066.7, 74763.92, 108797.44, 104276.56, 97934.3, 80059.92, 46038.68, 33212.34, 60732.54, 59619.82, 81971.48, 86468.76, 94281.44, 103242.1, 65876.1, 40066.6, 74289.4, 1.0]
[345450.7, 361252.08, 289508.24, 226070.96, 176549.86, 132572.2, 81697.4, 95015.84, 282869.54, 354485.26, 308799.08, 242370.7, 194502.16, 148201.74, 91290.82, 85597.98, 257138.78, 297530.4, 319165.08, 261020.1, 203028.3, 161722.52, 107380.5, 101131.6, 2.0]
"""
