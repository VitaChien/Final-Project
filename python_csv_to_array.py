import csv
import numpy as np
import matplotlib.pyplot as pyplot


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

class Portfolio():

    def __init__(self, csv):
        self.Mark = Markowiz(csv)
        self.Index = Index(csv)
        self.AverageReturn = AverageReturn()
        self.solve = Solve()

    def AverageReturn():

        average_return = []

        for i in range(1,len(name)-1):
            average_return.append(d[name[i]])
                
    def Markowitz(self):

        global Matrix_Markowitz

        Matrix = [d[name[1]]] 

        for i in range(2, len(name)):
            Matrix = np.vstack((Matrix, d[name[i]]))

        Matrix_Markowitz = np.cov(Matrix)

    def Index(self):

        beta_list = []    #beta for every stock
        e = []            #residual for every stock

        global average_return #average return for every stock
        average_return = [] 

        Rm = d[name[-1]]  

        A = np.vstack([Rm, np.ones(len(Rm))]).T 

        for i in range(1,len(name)-1):
            Rt = d[name[i]]
            beta, alpha = np.linalg.lstsq(A, Rt, rcond = -1)[0] 
            average_return.append(Rt) 
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
                         

    def solve(self, method, shortsale):

        global model 

        if method == 'M' and shortsale == True:
            model = 'Markowitz'
            solve_with_shortsale()

        if method == 'M' and shortsale == False:
            model = 'Markowitz'
            solve_without_shortsale()

        if method == 'I' and shortsle == True:
            model = 'Index'
            solve_with_shortsale()

        if method == 'I' and shortsale == False:
            model = 'Index'
            solve_without_shortsale()

def objective_normal(weight_list):

    Var_list = []
    if model == 'Markowitz':
        for i in range(6):
            Var_list.append(Matrix_Markowitz[i][i])

    if model == 'Index':
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
    for i in range(6):
        Var_P += Var_list[i] * weight_list[i]**2

        for j in range(6):
            if model == "Markowitz":
                Var_P += 2 * (weight_list[i] * weight_list[j] * Matrix_Markowitz[i][j])   
            if model == 'Index':
                 Var_P += 2 * (weight_list[i] * weight_list[j] * Matrix_Index[i][j]) 
    #目標式
    return Var_P

def objective_BMVP(weight_list):

    Var_list = []
    if model == "Markowitz":
        for i in range(6):
            Var_list.append(Matrix_Markowitz[i][i])

    if model == 'Index':
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
            if model == "Markowitz":
                Var_P += 2 * (weight_list[i] * weight_list[j] * Matrix_Markowitz[i][j])   
            if model == 'Index':
                 Var_P += 2 * (weight_list[i] * weight_list[j] * Matrix_Index[i][j])  

    SD_P = Var_P ** 0.5

    #目標式
    return (SD_P / ER)

def objective_GMVP(weight_list):

    Var_list = []
    if model == 'Markowitz':
        for i in range(6):
            Var_list.append(Matrix_Markowitz[i][i])

    if model == 'Index':
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
    for i in range(6):
        Var_P += Var_list[i] * weight_list[i]**2

        for j in range(6):
            if model == "Markowitz":
                Var_P += 2 * (weight_list[i] * weight_list[j] * Matrix_Markowitz[i][j])   
            if model == 'Index':
                 Var_P += 2 * (weight_list[i] * weight_list[j] * Matrix_Index[i][j]) 
    #目標式
    return Var_P

def constraint(weight_list): 
    sum_w = 1.0
    for i in range(6):
        sum_w = sum_w - weight_list[i]
    return sum_w

def constraint2_for_normal(weight_list):
    global required_return

    for i in range(6):
        required_return -= weight_list[i]*average_return[i]

def solve_with_shortsale(aims): 

    w0 = [1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0]

    con1 = {"type":"eq","fun":constraint}

    if aims == 'BMVP':
        sol = minimize(objective_BMVP, w0, method = "SLSQP", bounds = None, constraints = con1)
        
        weight = sol.x.tolist()
        
        for i in range(6):
            BMVP_return += weight[i]*average_return[i]

        BMVP_sd = BMVP_return*sol.fun #return的是(SD_P / ER)     

    if aims == 'GMVP':
        sol = minimize(objective_GMVP, w0, method = "SLSQP", bounds = None, constraints = con1)

        weight = sol.x.tolist()
        GMVP_sd = sol.fun**0.5  #return的是Var
        
        for i in range(6):
            GMVP_return += weight[i]*average_return[i]
  
    distance = (BMVP_return - GMVP_return)/6

    return_list = [GMVP_return]
    sd_list = [GMVP_sd]

    for i in range(6):
        point_return = BMVP_return + (i+1)*distance
        return_list.append(point_return)
    return_list.append(BMVP_return)    

    for i in range(6):
        required_return = return_list[i]
        sol = minimize(objective_normal, w0, method = "SLSQP", bounds = None, constraints = [con1,constraint2_for_normal])
        sd = sol.x
        sd_list.append(sd)    
    sd_list.append(BMVP_sd)    

def solve_without_shortsale(aims):

    w0 = [1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0]
    b = (0.0, 1.0) #bound
    bs = (b, b, b, b, b, b)
    con1 = {"type":"eq","fun":constraint}
    sol1 = minimize(objective_BMVP, w0, method = "SLSQP", bounds = bs, constraints = con1)
    er1 = 0
    er2 = 0 

    for i in range(len(sol1.x)):
        er1 += sol1.x[i] * average_return[i]

    sol2 = minimize(objective_GMVP, w0, method = "SLSQP", bounds = bs, constraints = con1)

    for i in range(len(sol2.x)):
        er2 += sol2.x[i] * average_return[i]

    gap = (er2 - er1)/6

    for i in range()

stock = '/Users/xuyuxiang/Desktop/test_data.csv'
mb64 = ""
d = dict()
data_len = -1

portfolio = Portfolio(csv_to_array(stock))

portfolio.solve(method = 'M', shortsale = True)


print(portfolio.Markowitz())
print(portfolio.Index())

print(data_len)

