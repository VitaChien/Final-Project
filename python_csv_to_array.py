import csv
import numpy as np
from scipy.optimize import minimize
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

    return None

class Portfolio():

    def __init__(self, i):
        self.Mark = self.Markowitz()
        self.Index = self.Index()
        self.AverageReturn = self.AverageReturn()

    def AverageReturn(self):

        global average_return
        average_return = []

        for i in range(1,len(name)):
            average_return.append(np.sum(d[name[i]])/60)
                
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

def constraint_for_normal_1(weight_list):

    return1 = return_list[1]

    for i in range(6):
        return1 -= weight_list[i]*average_return[i]

    return return1

def constraint_for_normal_2(weight_list):

    return2 = return_list[2]

    for i in range(6):
        return2 -= weight_list[i]*average_return[i]

    return return2  

def constraint_for_normal_3(weight_list):

    return3 = return_list[3]

    for i in range(6):
        return3 -= weight_list[i]*average_return[i]

    return return3 

def constraint_for_normal_4(weight_list):

    return4 = return_list[4]
    for i in range(6):
        return4 -= weight_list[i]*average_return[i]

    return  return4

def constraint_for_normal_5(weight_list):

    return5 = return_list[5]

    for i in range(6):
        return5 -= weight_list[i]*average_return[i] 

    return return5                        

def solve_with_shortsale(): 

    w0 = [1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0]

    global CON

    CON = {"type":"eq","fun":constraint}

    BMVP_return = 0
    GMVP_return = 0

    sol1 = minimize(objective_BMVP, w0, method = "SLSQP", bounds = None, constraints = CON)
        
    weight = sol1.x.tolist()
        
    for i in range(6):
        BMVP_return += weight[i]*average_return[i]

    BMVP_sd = BMVP_return*sol1.fun #return的是(SD_P / ER)     


    sol2 = minimize(objective_GMVP, w0, method = "SLSQP", bounds = None, constraints = CON)

    weight = sol2.x.tolist()
    GMVP_sd = sol2.fun**0.5  #return的是Var
        
    for i in range(6):
        GMVP_return += weight[i]*average_return[i]
  
    distance = (BMVP_return - GMVP_return)/6

    global return_list

    return_list = [GMVP_return]
    sd_list = [GMVP_sd]

    point_return = GMVP_return

    for i in range(5):
        point_return += distance
        return_list.append(point_return)
    return_list.append(BMVP_return)    

    con1 = {"type":"eq","fun":constraint_for_normal_1}
    con2 = {"type":"eq","fun":constraint_for_normal_2}
    con3 = {"type":"eq","fun":constraint_for_normal_3}
    con4 = {"type":"eq","fun":constraint_for_normal_4}
    con5 = {"type":"eq","fun":constraint_for_normal_5}


    con_list = [con1, con2, con3, con4, con5]

    for i in range(5):

        sol = minimize(objective_normal, w0, method = "SLSQP", bounds = None, constraints = [CON, con_list[i]])
        sd = sol.fun**0.5
        sd_list.append(sd)    
    sd_list.append(BMVP_sd) 

    print(return_list)
    print(sd_list)   

def solve_without_shortsale(): 

    w0 = [1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0]

    b = (0.0, 1.0)
    bs = (b,b,b,b,b)

    global CON

    CON = {"type":"eq","fun":constraint}

    BMVP_return = 0
    GMVP_return = 0

    sol1 = minimize(objective_BMVP, w0, method = "SLSQP", bounds = bs, constraints = CON)
        
    weight = sol1.x.tolist()
        
    for i in range(6):
        BMVP_return += weight[i]*average_return[i]

    BMVP_sd = BMVP_return*sol1.fun #return的是(SD_P / ER)     


    sol2 = minimize(objective_GMVP, w0, method = "SLSQP", bounds = bs, constraints = CON)

    weight = sol2.x.tolist()
    GMVP_sd = sol2.fun**0.5  #return的是Var
        
    for i in range(6):
        GMVP_return += weight[i]*average_return[i]
  
    distance = (BMVP_return - GMVP_return)/6

    global return_list

    return_list = [GMVP_return]
    sd_list = [GMVP_sd]

    point_return = GMVP_return

    for i in range(5):
        point_return += distance
        return_list.append(point_return)
    return_list.append(BMVP_return)    

    con1 = {"type":"eq","fun":constraint_for_normal_1}
    con2 = {"type":"eq","fun":constraint_for_normal_2}
    con3 = {"type":"eq","fun":constraint_for_normal_3}
    con4 = {"type":"eq","fun":constraint_for_normal_4}
    con5 = {"type":"eq","fun":constraint_for_normal_5}


    con_list = [con1, con2, con3, con4, con5]

    for i in range(5):

        sol = minimize(objective_normal, w0, method = "SLSQP", bounds = None, constraints = [CON, con_list[i]])
        sd = sol.fun**0.5
        sd_list.append(sd)    
    sd_list.append(BMVP_sd) 

    print(return_list)
    print(sd_list)       

#畫圖
def draw(return_list, sd_list):
    x = return_list
    y = sd_list
    pyplot.plot(x, y)
    pyplot.show()


stock = '/Users/xuyuxiang/Desktop/test_data.csv'
mb64 = ""
d = dict()
data_len = -1

portfolio = Portfolio(csv_to_array(stock))

portfolio.solve(method = 'M', shortsale = True)