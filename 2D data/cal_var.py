import csv
import math
import operator
import numpy as np
from decimal import Decimal

count = 0
list_var = []
var =0
j = 0
with open('/home/mahima/te.txt', 'r') as csvfile:
	lines = csvfile.read()
	t = lines.split(',')
	for i in range(25):
		for j in range(i, len(t), 25):
			#print(t[j])
			count += int(t[j])
		mean = float(count)/50.0
		for j in range(i, len(t), 24):
			var += ((int(t[i])-float(mean))**2)/50.0
			#print(var)
		list_var.append(var)
		count = 0
		var = 0
		i += 1
	print(list_var)

"""
368907, 395409, 289508, 161285, 95095, 54168, 51320, 122881, 132888, 276319, 324929, 272794, 195509, 128639, 95026, 112469, 146322, 281219, 301103, 263315, 199861, 137878, 101388, 107487, 0
72180, 113147, 108597, 110749, 64599, 49433, 32040, 55066, 74763, 108797, 104276, 97934, 80059, 46038, 33212, 60732, 59619, 81971, 86468, 94281, 103242, 65876, 40066, 74289, 1
345450, 361252, 289508, 226070, 176549, 132572, 81697, 95015, 282869, 354485, 308799, 242370, 194502, 148201, 91290, 85597, 257138, 297530, 319165, 261020, 203028, 161722, 107380, 101131, 2

"""

