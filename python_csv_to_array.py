import csv
import numpy as np
import pulp 

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

    global Matrix_Index
    
    Matrix_Index = [[] for x in range(len(name) - 1)]

    for i in range(len(name)-1):
        for j in range(len(name)-1):
            if i != j:
                Matrix_Index[i].append(beta_list[i] * beta_list[j] * Rm_Var)
            else:
                Matrix_Index[i].append(beta_list[i]**2 * Rm_Var + e_Var)    

    print(Matrix_Index)  	


def objective(weight_list):

    Var_list = []

    for i in range(6):
        Var_list.append(Matrix_Index[i][i])

    w1 = weight_list[0]
    w2 = weight_list[1]
    w3 = weight_list[2]
    w4 = weight_list[3]
    w5 = weight_list[4]
    w6 = weight_list[5]

    weight_list = [w1, w2, w3, w4, w5, w6]

    #計算portfolio的expected return 和 standard deviation
    Var_P = 0
    ER = 0
    for i in range(6):
        Var_P += Var_list[i] * weight_list[i]**2
        ER += weight_list[i] * sum(d[name[i+1]]) / len(d[name[i+1]])

        for j in range(6):
            Var_P += 2 * (weight_list[i] * weight_list[j] * Matrix_Index[i][j])    

    SD_P = Var_P ** 0.5

    #目標式
    return (SD_P / ER)


def constraint(weight_list): 
    sum_w = 1.0
    for i in range(6):
        sum_w = sum_w - weight_list[i]
    return sum_w


def solve():

    w0 = [1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0]

    b = (0.0, 1.0) #bound
    bs = (b, b, b, b, b, b)

    con1 = {"type":"eq","fun":constraint}

    sol = minimize(objective, w0, method = "SLSQP", bounds = bs, constraints = con1)

    print(sol) 

stock = '/Users/xuyuxiang/Desktop/test_data.csv'
mb64 = ""
d = dict()
data_len = -1

csv_to_array(stock)

array_to_CovMatrix_Markowitz()
array_to_CovMatrix_Index()

print(data_len)

