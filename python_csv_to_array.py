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


stock = '/Users/xuyuxiang/Desktop/test_data.csv'
mb64 = ""
d = dict()
data_len = -1

csv_to_array(stock)



print(name)
print(d)

print(data_len)

