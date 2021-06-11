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
def main():
	data=[]
	mean=[]
	var1 = np.zeros((2,2), dtype=float)
	var2 = np.zeros((2,2), dtype=float)
	var3 = np.zeros((2,2), dtype=float)
	var_atr = np.zeros((2,2), dtype=float)
	var_gen = np.zeros((2,2), dtype=float)
	var_obs = np.zeros((2,2), dtype=float)
	mean1 = [[0,0,0],[0,0,0],[0,0,0]] 
	mya = 0
	myb = 0
	myc = 0
	with open('/home/mahima/LSdata/mean_var_ls_data.txt', 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(len(dataset)):
			#for y in range(2):
			#	dataset[x][y] = float(dataset[x][y])
	        #if random.random() < split:
			mean.append(dataset[x])
	       # else:
	          #  testSet.append(dataset[x])
	with open('/home/mahima/LSdata/var_ls_train.csv', 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(len(dataset)):
			#for y in range(2):
			#	dataset[x][y] = dataset[x][y]
	       # if random.random() < split:
	           # trainingSet.append(dataset[x])
	        #else:
			data.append(dataset[x])
	for col in range(0,2):
		for row in range(len(data)):
			if data[row][-1] =='a':
				
				var1[col][col] += pow((float(data[row][col]) - float(mean[0][col])), 2)
				mya = mya +1
				#print(var1[col][col])
			elif data[row][-1] =='b':
				myb = myb + 1
				var2[col][col] += pow((float(data[row][col]) - float(mean[1][col])), 2)
			elif data[row][-1] =='c':
				myc  = myc +1
				var3[col][col] += pow((float(data[row][col]) - float(mean[2][col])), 2)

	#print('var1:')
	#print(mya/2)
	#print('var2:')
	#print(myb/2)
	#print('var3:')
	#print(myc/2)
	mya = 2/mya
	myb = 2/myb
	myc = 2/myc
	var_atr = np.multiply(var1, mya)
	var_gen = np.multiply(var2, myb)
	var_obs = np.multiply(var3, myc)
	#print("var3")	
	#print(var_obs)
	var_atr = np.linalg.inv(var_atr) 
	var_gen = np.linalg.inv(var_gen) 
	var_obs = np.linalg.inv(var_obs) 
	for col in range(0,2):
		mean1[0][col]=float(mean[0][col])*var_atr[col][col]
		mean1[1][col]=float(mean[1][col])*var_atr[col][col]
		mean1[2][col]=float(mean[2][col])*var_atr[col][col]

	mean1[0][2] = 1
	mean1[1][2] = 2
	mean1[2][2] = 3
	

	#mean1[0][0] = float(mean[0][0])*var_atr 
	#mean1[1][0] = float(mean[0][0])*var_gen
	#mean1[2][0] = float(mean[0][0])*var_obs

	#mean1[0][1] = float(mean[0][1])*var_atr 
	#mean1[1][1] = float(mean[0][1])*var_gen
	#mean1[2][1] = float(mean[0][1])*var_obs

	print(mean1[0][0])
	with open("/home/mahima/Desktop/mean2.txt", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(mean1)
main()
