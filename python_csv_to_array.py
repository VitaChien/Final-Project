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

def array_to_CovMatrix_Index():

    beta_list = []    #beta for every stock
    e = []            #residual for every stock

    Rm = d[name[-1]]  

    A = np.vstack([Rm, np.ones(len(Rm))]).T 

    for i in range(1,len(name)-1):
        Rt = d[name[i]]
        beta, alpha = np.linalg.lstsq(A, Rt, rcond = -1)[0]  
        beta_list.append(beta)

        residual = Rt - (alpha + beta * Rm)
        e.append(residual)

    beta_list.append(1.0)    
    e = np.asarray(e)   
    e_Var = np.var(e)     #residual variance

    Rm = np.asarray(Rm) 
    Rm_Var = np.var(Rm)   #market return variance
    
    Matrix_Index = [[] for x in range(len(name)-1)]

    for i in range(len(name)-1):
        for j in range(len(name)-1):
            if i != j:
                Matrix_Index[i].append(beta_list[i] * beta_list[j] * Rm_Var)
            else:
                Matrix_Index[i].append(beta_list[i]**2 * Rm_Var + e_Var)    

    print(Matrix_Index)  	


stock = '/Users/xuyuxiang/Desktop/test_data.csv'
mb64 = ""
d = dict()
data_len = -1

csv_to_array(stock)

array_to_CovMatrix_Markowitz()
array_to_CovMatrix_Index()

print(data_len)

