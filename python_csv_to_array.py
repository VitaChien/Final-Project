#Project csv 資料處理

import csv
import numpy as np

def csv_to_array(stock):

	global data_len, d, name

	fh1 = open(stock, 'r', newline = '')
	csv1 = csv.DictReader(fh1)
	name = csv1.fieldnames

	for aline in csv1: #converting csv to dictionary of arrays

		data_len += 1
 
		for i in range(1, len(name)):

			try:
				d[name[i]] = np.append(d[name[i]], [float(aline[name[i]]) ])

			except KeyError:
				d[name[i]] = np.array([])

	fh1.close()

def array_to_CovMatrix_Markowitz():

	global Matrix

	Matrix = [d[name[1]]]

	for i in range(2, len(name)):
		Matrix = np.vstack((Matrix, d[name[i]]))

	print(np.cov(Matrix))

stock = '/Users/xuyuxiang/Desktop/test_data.csv'
mb64 = ""
d = dict()
data_len = -1

csv_to_array(stock)

array_to_CovMatrix_Markowitz()

print(data_len)

