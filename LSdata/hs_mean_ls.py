import csv
import random
import math
import operator
import csv
from pandas import *

	
def main():
	#cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.60
	#loadDataset('/home/megha/Downloads/montu lapi/MeghaInternship/Megha/group04 copy.csv', split, trainingSet, testSet)
	#with open('/home/megha/Desktop/separable/LS_Group04/Class1.csv', 'r') as csvfile:
	with open('/home/mahima/LSdata/Class1.csv', 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(len(dataset)):
			for y in range(2):
				dataset[x][y] = dataset[x][y]
			if random.random() < split:
				trainingSet.append(dataset[x])
			else:
				testSet.append(dataset[x])
	with open('/home/mahima/LSdata/Class2.csv', 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(len(dataset)):
			for y in range(2):
				dataset[x][y] = dataset[x][y]
			if random.random() < split:
				trainingSet.append(dataset[x])
			else:
				testSet.append(dataset[x])
	with open('/home/mahima/LSdata/Class3.csv', 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(len(dataset)):
			for y in range(2):
				dataset[x][y] = dataset[x][y]
			if random.random() < split:
				trainingSet.append(dataset[x])
			else:
				testSet.append(dataset[x])				
#print(testSet)
	#with open('/home/megha/Desktop/test.data', 'r') as csvfile:
	#	lines = csv.reader(csvfile)
	#	dataset = list(lines)
	#	for x in range(len(dataset)):
	#		for y in range(4):
	#			dataset[x][y] = float(dataset[x][y])
	       # if random.random() < split:
	           # trainingSet.append(dataset[x])
	        #else:
	#		testSet.append(dataset[x])
	#print('Train set: ' + repr(len(trainingSet)))
	#print 'Test set: ' + repr(len(testSet))
	# generate predictions
	mean = [[0,0,0],[0,0,0],[0,0,0]] 
	mean[0][2] = 'a'
	mean[1][2] = 'b'
	mean[2][2] = 'c'
	for col in range(0,2):
		a = 0
		b = 0
		c = 0
		count1 = 0
		count2 = 0
		count3 = 0
		for row in range(len(trainingSet)):
			if y == 2:
				break
			if trainingSet[row][-1] =='a':
			#if row >= 0 and row < :
				a = a + float(trainingSet[row][col])
				count1 = count1 +1
				#print(a)
			#if row >= 50 and row < 100:
			if trainingSet[row][-1] =='b':
				b = b + float(trainingSet[row][col])
				count2 = count2+1
				#print(b)
			#if row >= 100:
			if trainingSet[row][-1] =='c':
				c = c + float(trainingSet[row][col])
				count3 = count3+1
				#print(c)
		mean[0][col] = float(a)/500.0 
		mean[1][col] = float(b)/500.0 
		mean[2][col] = float(c)/500.0 
	print(count1)
	print(count2)
	print(count3)

	with open("/home/mahima/LSdata/mean_var_ls.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(mean)
	with open("/home/mahima/LSdata/var_ls_test.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(testSet)
	with open("/home/mahima/LSdata/var_ls_train.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(trainingSet)


main()

