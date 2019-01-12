import csv
import numpy as np
from scipy.optimize import minimize
import tkinter as tk
import tkinter.font as tkFont
import math
from PIL import ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


master = tk.Tk()
master.geometry('900x650')

class mainpage(tk.Frame):
  
  def __init__(self):
    tk.Frame.__init__(self) 
    self.pack()
    self.createWidgets()

  def createWidgets(self):

    f2 = tkFont.Font(size = 32, family = "Courier New")

    self.imageSqrt = ImageTk.PhotoImage(file = "/Users/xuyuxiang/Desktop/image/螢幕快照 2019-01-06 下午4.26.47.png")
    self.btn1 = tk.Label(self, image = self.imageSqrt) 
    self.imageSqrt2 = ImageTk.PhotoImage(file = "/Users/xuyuxiang/Desktop/image/螢幕快照 2019-01-06 下午4.08.30.png")
    self.btn2 = tk.Button(self, image = self.imageSqrt2, command = self.secpage)

    self.btn1.grid(row = 0, column = 0, pady = 30)
    self.btn2.grid(row = 1, column = 0, pady = 30)

  def secpage(self):
    self.btn1.destroy()
    self.btn2.destroy()

    f2 = tkFont.Font(size = 32, family = "Courier New")
    self.imageSqrt = ImageTk.PhotoImage(file = "/Users/xuyuxiang/Desktop/image/images.png")
    self.btn1 = tk.Label(self, image = self.imageSqrt)
    self.imageSqrt1 = ImageTk.PhotoImage(file = "/Users/xuyuxiang/Desktop/image/螢幕快照 2019-01-06 下午4.35.51.png")
    self.check1 = tk.Button(self, image = self.imageSqrt1, width = 340, command = self.thirdpage)
    self.explain1 = tk.Label(self, text = "用自己的歷史作預測", height = 1, width = 20, font = f2)
    self.imageSqrt2= ImageTk.PhotoImage(file = "/Users/xuyuxiang/Desktop/image/螢幕快照 2019-01-06 下午4.36.13.png")
    self.check2 = tk.Button(self, image = self.imageSqrt2, width = 340, command = self.thirdpage)
    self.explain2 = tk.Label(self, text = "用大盤的歷史作預測", height = 1, width = 20, font = f2)
    self.imageSqrt3 = ImageTk.PhotoImage(file = "/Users/xuyuxiang/Desktop/image/螢幕快照 2019-01-06 下午4.35.33.png")
    self.check3 = tk.Button(self, image = self.imageSqrt3, width = 340, command = self.thirdpage)
    self.explain3 = tk.Label(self, text = "還要我解釋ㄇ", height = 1, width = 20, font = f2)

    self.btn1.grid(row = 0, column = 0, columnspan = 2, pady = 50)
    self.check1.grid(row = 1, column = 0, pady = 10)
    self.explain1.grid(row = 1, column = 1, pady = 10)
    self.check2.grid(row = 2, column = 0, pady = 10)
    self.explain2.grid(row = 2, column = 1, pady = 10)
    self.check3.grid(row = 3, column = 0, pady = 10)
    self.explain3.grid(row = 3, column = 1, pady = 10)


  def thirdpage(self):

    self.btn1.destroy()
    self.check1.destroy()
    self.check2.destroy()
    self.check3.destroy()
    self.explain1.destroy()
    self.explain2.destroy()
    self.explain3.destroy()

    f1 = tkFont.Font(size = 48, family = "Courier New")
    f2 = tkFont.Font(size = 32, family = "Courier New")
    
    self.imageSqrt = ImageTk.PhotoImage(file = "/Users/xuyuxiang/Desktop/image/螢幕快照 2019-01-08 下午11.54.32.png")
    self.btn1 = tk.Label(self, image = self.imageSqrt)
    self.entry1 = tk.Entry(self, width = 50)

    self.imageSqrt2 = ImageTk.PhotoImage(file = "/Users/xuyuxiang/Desktop/image/螢幕快照 2019-01-08 下午11.52.02.png")
    self.btn2 = tk.Button(self, image = self.imageSqrt2, command = self.fourpage)
    self.btn1.pack(side = "top", pady=30)
    self.entry1.pack(side = "top", pady=30)
    self.btn2.pack(side = "bottom", pady=70)

  def fourpage(self):
    global name
    self.btn1.destroy()
    self.entry1.destroy()
    self.btn2.destroy()
    f1 = tkFont.Font(size = 28, family = "Courier New")
    f2 = tkFont.Font(size = 32, family = "Courier New")
    
    self.imageSqrt1 = ImageTk.PhotoImage(file = "/Users/xuyuxiang/Desktop/image/螢幕快照 2019-01-09 上午1.01.24.png")
    self.btn1 = tk.Label(self, image = self.imageSqrt1)
    self.imageSqrt2 = ImageTk.PhotoImage(file = "/Users/xuyuxiang/Desktop/image/螢幕快照 2019-01-09 上午1.02.05.png")
    self.re = tk.Label(self, image = self.imageSqrt2)
    self.btn4 = tk.Label(self, text = name[1], height = 1, width = 10, font = f2)
    self.re1 = tk.Entry(self, width = 10)
    self.repa1 = tk.Label(self, text = "%", height = 1, width = 2, font = f2)
    self.btn7 = tk.Label(self, text = name[2], height = 1, width = 10, font = f2)
    self.re2 = tk.Entry(self, width = 10)
    self.repa2 = tk.Label(self, text = "%", height = 1, width = 2, font = f2)  
    self.btn10 = tk.Label(self, text = name[3], height = 1, width = 10, font = f2)
    self.re3 = tk.Entry(self, width = 10)
    self.repa3 = tk.Label(self, text = "%", height = 1, width = 2, font = f2) 
    self.btn13 = tk.Label(self, text = name[4], height = 1, width = 10, font = f2)
    self.re4 = tk.Entry(self, width = 10)
    self.repa4 = tk.Label(self, text = "%", height = 1, width = 2, font = f2)
    self.btn16 = tk.Label(self, text = name[5], height = 1, width = 10, font = f2)
    self.re5 = tk.Entry(self, width = 10)
    self.repa5 = tk.Label(self, text = "%", height = 1, width = 2, font = f2)
    self.btn19 = tk.Label(self, text = name[6], height = 1, width = 10, font = f2)
    self.re6 = tk.Entry(self, width = 10)
    self.repa6 = tk.Label(self, text = "%", height = 1, width = 2, font = f2)
    self.btn22 = tk.Label(self, text = name[7], height = 1, width = 10, font = f2)
    self.re7 = tk.Entry(self, width = 10)
    self.repa7 = tk.Label(self, text = "%", height = 1, width = 2, font = f2)
    self.btn25 = tk.Label(self, text = name[8], height = 1, width = 10, font = f2)
    self.re8 = tk.Entry(self, width = 10)
    self.repa8 = tk.Label(self, text = "%", height = 1, width = 2, font = f2)
    self.btn28 = tk.Label(self, text = name[9], height = 1, width = 10, font = f2)
    self.re9 = tk.Entry(self, width = 10)
    self.repa9 = tk.Label(self, text = "%", height = 1, width = 2, font = f2)
    self.btn31 = tk.Label(self, text = name[10], height = 1, width = 10, font = f2)
    self.re10 = tk.Entry(self, width = 10)
    self.repa10 = tk.Label(self, text = "%", height = 1, width = 2, font = f2)
    self.imageSqrt = ImageTk.PhotoImage(file = "/Users/xuyuxiang/Desktop/image/螢幕快照 2019-01-08 下午11.52.02.png")
    self.btn34 = tk.Button(self, image = self.imageSqrt, command = self.fivepage)
    
    self.btn1.grid(row = 0, column = 0, pady=20)
    self.re.grid(row = 0, column = 1, columnspan = 2, pady=20)
    self.btn4.grid(row = 1, column = 0, pady=1)
    self.re1.grid(row = 1, column = 1, pady = 1)
    self.repa1.grid(row = 1, column = 2, pady = 1, sticky = tk.NE + tk.SW)
    self.btn7.grid(row = 2, column = 0, pady=1)
    self.re2.grid(row = 2, column = 1, pady = 1)
    self.repa2.grid(row = 2, column = 2, pady = 1, sticky = tk.NE + tk.SW)
    self.btn10.grid(row = 3, column = 0, pady=1)
    self.re3.grid(row = 3, column = 1, pady = 1)
    self.repa3.grid(row = 3, column = 2, pady = 1, sticky = tk.NE + tk.SW)
    self.btn13.grid(row = 4, column = 0, pady=1)
    self.re4.grid(row = 4, column = 1, pady = 1)
    self.repa4.grid(row = 4, column = 2, pady = 1, sticky = tk.NE + tk.SW)
    self.btn16.grid(row = 5, column = 0, pady=1)
    self.re5.grid(row = 5, column = 1, pady = 1)
    self.repa5.grid(row = 5, column = 2, pady = 1, sticky = tk.NE + tk.SW)
    self.btn19.grid(row = 6, column = 0, pady=1)
    self.re6.grid(row = 6, column = 1, pady = 1)
    self.repa6.grid(row = 6, column = 2, pady = 1, sticky = tk.NE + tk.SW)
    self.btn22.grid(row = 7, column = 0, pady=1)
    self.re7.grid(row = 7, column = 1, pady = 1)
    self.repa7.grid(row = 7, column = 2, pady = 1, sticky = tk.NE + tk.SW)
    self.btn25.grid(row = 8, column = 0, pady=1)
    self.re8.grid(row = 8, column = 1, pady = 1)
    self.repa8.grid(row = 8, column = 2, pady = 1, sticky = tk.NE + tk.SW)
    self.btn28.grid(row = 9, column = 0, pady=1)
    self.re9.grid(row = 9, column = 1, pady = 1)
    self.repa9.grid(row = 9, column = 2, pady = 1, sticky = tk.NE + tk.SW)
    self.btn31.grid(row = 10, column = 0, pady=1)
    self.re10.grid(row = 10, column = 1, pady = 1)
    self.repa10.grid(row = 10, column = 2, pady = 1, sticky = tk.NE + tk.SW)
    self.btn34.grid(row = 11, column = 1, pady=1)
  
  def fivepage(self) :
    global name2
    self.btn1.destroy()
    self.btn4.destroy()
    self.btn7.destroy()
    self.btn10.destroy()
    self.btn13.destroy()
    self.btn16.destroy()
    self.btn19.destroy()
    self.btn22.destroy()
    self.btn25.destroy()
    self.btn28.destroy()
    self.btn31.destroy()
    self.btn34.destroy()
    self.re.destroy()
    self.re1.destroy()
    self.re2.destroy()
    self.re3.destroy()
    self.re4.destroy()
    self.re5.destroy()
    self.re6.destroy()
    self.re7.destroy()
    self.re8.destroy()
    self.re9.destroy()
    self.re10.destroy()
    self.repa1.destroy()
    self.repa2.destroy()
    self.repa3.destroy()
    self.repa4.destroy()
    self.repa5.destroy()
    self.repa6.destroy()
    self.repa7.destroy()
    self.repa8.destroy()
    self.repa9.destroy()
    self.repa10.destroy()



    f2 = tkFont.Font(size = 28, family = "Courier New")

    self.imageSqrtpic = ImageTk.PhotoImage(file = "/Users/xuyuxiang/Desktop/image/the_figure.png")
    self.pic = tk.Label(self, image = self.imageSqrtpic)
    self.imageSqrt = ImageTk.PhotoImage(file = "/Users/xuyuxiang/Desktop/image/螢幕快照 2019-01-09 下午12.52.37.png")
    self.btn1 = tk.Label(self, image = self.imageSqrt)
    self.imageSqrt1 = ImageTk.PhotoImage(file = "/Users/xuyuxiang/Desktop/image/螢幕快照 2019-01-09 下午12.52.11.png")
    self.btn2 = tk.Label(self, image = self.imageSqrt1)
    self.blank1 = tk.Label(self, height = 1, width = 2)
    self.imageSqrt2 = ImageTk.PhotoImage(file = "/Users/xuyuxiang/Desktop/image/螢幕快照 2019-01-09 下午12.52.37.png")
    self.btn3 = tk.Label(self, image = self.imageSqrt2)
    self.imageSqrt3 = ImageTk.PhotoImage(file = "/Users/xuyuxiang/Desktop/image/螢幕快照 2019-01-09 下午12.52.11.png")
    self.btn4 = tk.Label(self, image = self.imageSqrt3)
    self.btn5 = tk.Label(self, text = name2[1], height = 1, width = 8, font = f2)
    self.btn6 = tk.Label(self, text = "0.1717", height = 1, width = 8, font = f2)
    self.blank2 = tk.Label(self, height = 1, width = 2)
    self.btn7 = tk.Label(self, text = name2[2], height = 1, width = 8, font = f2)
    self.btn8 = tk.Label(self, text = "0.0673", height = 1, width = 8, font = f2)
    self.btn9 = tk.Label(self, text = name2[3], height = 1, width = 8, font = f2)
    self.btn10 = tk.Label(self, text = "-0.0687", height = 1, width = 8, font = f2)
    self.blank3 = tk.Label(self, height = 1, width = 2)
    self.btn11 = tk.Label(self, text = name2[4], height = 1, width = 8, font = f2)
    self.btn12 = tk.Label(self, text = "0.0037", height = 1, width = 8, font = f2)
    self.btn13 = tk.Label(self, text = name2[5], height = 1, width = 8, font = f2)
    self.btn14 = tk.Label(self, text = "-0.0480", height = 1, width = 8, font = f2)
    self.blank4 = tk.Label(self, height = 1, width = 2)
    self.btn15 = tk.Label(self, text = name2[6], height = 1, width = 8, font = f2)
    self.btn16 = tk.Label(self, text = "-0.0547", height = 1, width = 8, font = f2)
    self.btn17 = tk.Label(self, text = name2[7], height = 1, width = 8, font = f2)
    self.btn18 = tk.Label(self, text = "0.1271", height = 1, width = 8, font = f2)
    self.blank5 = tk.Label(self, height = 1, width = 2)
    self.btn19 = tk.Label(self, text = name2[8], height = 1, width = 8, font = f2)
    self.btn20 = tk.Label(self, text = "0.6793", height = 1, width = 8, font = f2)
    self.btn21 = tk.Label(self, text = name2[9], height = 1, width = 8, font = f2)
    self.btn22 = tk.Label(self, text = "-0.0324", height = 1, width = 8, font = f2)
    self.blank6 = tk.Label(self, height = 1, width = 2)
    self.btn23 = tk.Label(self, text = name2[10], height = 1, width = 8, font = f2)
    self.btn24 = tk.Label(self, text = "0.1547", height = 1, width = 8, font = f2)
   

    self.pic.grid(row = 0, column = 0, columnspan = 5, pady=20)
    self.btn1.grid(row = 1, column = 0, pady=1)
    self.btn2.grid(row = 1, column = 1, pady=1)
    self.blank1.grid(row = 1, column = 2, pady=1)
    self.btn3.grid(row = 1, column = 3, pady=1)
    self.btn4.grid(row = 1, column = 4, pady=1)
    self.btn5.grid(row = 2, column = 0, pady=1)
    self.btn6.grid(row = 2, column = 1, pady=1)
    self.blank2.grid(row = 2, column = 2, pady=1)
    self.btn7.grid(row = 2, column = 3, pady=1)
    self.btn8.grid(row = 2, column = 4, pady=1)
    self.btn9.grid(row = 3, column = 0, pady=1)
    self.btn10.grid(row = 3, column = 1, pady=1)
    self.blank3.grid(row = 3, column = 2, pady=1)
    self.btn11.grid(row = 3, column = 3, pady=1)
    self.btn12.grid(row = 3, column = 4, pady=1)
    self.btn13.grid(row = 4, column = 0, pady=1)
    self.btn14.grid(row = 4, column = 1, pady=1)
    self.blank4.grid(row = 4, column = 2, pady=1)
    self.btn15.grid(row = 4, column = 3, pady=1)
    self.btn16.grid(row = 4, column = 4, pady=1)
    self.btn17.grid(row = 5, column = 0, pady=1)
    self.btn18.grid(row = 5, column = 1, pady=1)
    self.blank5.grid(row = 5, column = 2, pady=1)
    self.btn19.grid(row = 5, column = 3, pady=1)
    self.btn20.grid(row = 5, column = 4, pady=1)
    self.btn21.grid(row = 6, column = 0, pady=1)
    self.btn22.grid(row = 6, column = 1, pady=1)
    self.blank6.grid(row = 6, column = 2, pady=1)
    self.btn23.grid(row = 6, column = 3, pady=1)
    self.btn24.grid(row = 6, column = 4, pady=1)


def csv_to_array(stock):

    global data_len, d, name, name2

    fh1 = open(stock, 'r', newline = '')
    csv1 = csv.DictReader(fh1)
    name = csv1.fieldnames
    name2 = ['0']
    for i in range(1, 11):
        name3 = name[i].split()
        name2.append(name3[1])

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

        if method == 'I' and shortsale == True:
            model = 'Index'
            solve_with_shortsale()

        if method == 'I' and shortsale == False:
            model = 'Index'
            solve_without_shortsale()

def objective_normal(weight_list):

    Var_list = []
    if model == 'Markowitz':
        for i in range(10):
            Var_list.append(Matrix_Markowitz[i][i])

    if model == 'Index':
        for i in range(10):
            Var_list.append(Matrix_Index[i][i])

    w1 = weight_list[0]
    w2 = weight_list[1]
    w3 = weight_list[2]
    w4 = weight_list[3]
    w5 = weight_list[4]
    w6 = weight_list[5]
    w7 = weight_list[6]
    w8 = weight_list[7]
    w9 = weight_list[8]
    w10 = weight_list[9]

    weight_list = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10]

    #計算portfolio的expected return 和 standard deviation
    Var_P = 0
    for i in range(10):
        Var_P += Var_list[i] * weight_list[i]**2

        for j in range(10):
            if model == "Markowitz":
                Var_P += 2 * (weight_list[i] * weight_list[j] * Matrix_Markowitz[i][j])   
            if model == 'Index':
                 Var_P += 2 * (weight_list[i] * weight_list[j] * Matrix_Index[i][j]) 
    #目標式
    return Var_P

def objective_BMVP(weight_list):

    Var_list = []
    if model == "Markowitz":
        for i in range(10):
            Var_list.append(Matrix_Markowitz[i][i])

    if model == 'Index':
        for i in range(10):
            Var_list.append(Matrix_Index[i][i])

    w1 = weight_list[0]
    w2 = weight_list[1]
    w3 = weight_list[2]
    w4 = weight_list[3]
    w5 = weight_list[4]
    w6 = weight_list[5]
    w7 = weight_list[6]
    w8 = weight_list[7]
    w9 = weight_list[8]
    w10 = weight_list[9]

    weight_list = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10]

    #計算portfolio的expected return 和 standard deviation
    Var_P = 0
    ER = 0
    for i in range(10):
        Var_P += Var_list[i] * weight_list[i]**2
        ER += weight_list[i] * sum(d[name[i+1]]) / len(d[name[i+1]])

        for j in range(10):
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
        for i in range(10):
            Var_list.append(Matrix_Markowitz[i][i])

    if model == 'Index':
        for i in range(10):
            Var_list.append(Matrix_Index[i][i])

    w1 = weight_list[0]
    w2 = weight_list[1]
    w3 = weight_list[2]
    w4 = weight_list[3]
    w5 = weight_list[4]
    w6 = weight_list[5]
    w7 = weight_list[6]
    w8 = weight_list[7]
    w9 = weight_list[8]
    w10 = weight_list[9]

    weight_list = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10]

    #計算portfolio的expected return 和 standard deviation
    Var_P = 0
    for i in range(10):
        Var_P += Var_list[i] * weight_list[i]**2

        for j in range(10):
            if model == "Markowitz":
                Var_P += 2 * (weight_list[i] * weight_list[j] * Matrix_Markowitz[i][j])   
            if model == 'Index':
                 Var_P += 2 * (weight_list[i] * weight_list[j] * Matrix_Index[i][j]) 
    #目標式
    return Var_P

def constraint(weight_list): 
    sum_w = 1.0
    for i in range(10):
        sum_w = sum_w - weight_list[i]
    return sum_w

def constraint_for_normal_1(weight_list):

    return1 = return_list[1]

    for i in range(10):
        return1 -= weight_list[i]*average_return[i]

    return return1

def constraint_for_normal_2(weight_list):

    return2 = return_list[2]

    for i in range(10):
        return2 -= weight_list[i]*average_return[i]

    return return2  

def constraint_for_normal_3(weight_list):

    return3 = return_list[3]

    for i in range(10):
        return3 -= weight_list[i]*average_return[i]

    return return3 

def constraint_for_normal_4(weight_list):

    return4 = return_list[4]
    for i in range(10):
        return4 -= weight_list[i]*average_return[i]

    return  return4

def constraint_for_normal_5(weight_list):

    return5 = return_list[5]

    for i in range(10):
        return5 -= weight_list[i]*average_return[i] 

    return return5                        

def solve_with_shortsale(): 

    w0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    global CON, answerx, answery

    CON = {"type":"eq","fun":constraint}

    BMVP_return = 0
    GMVP_return = 0

    sol1 = minimize(objective_BMVP, w0, method = "SLSQP", bounds = None, constraints = CON)
        
    weight = sol1.x.tolist()
        
    for i in range(10):
        BMVP_return += weight[i]*average_return[i]

    BMVP_sd = BMVP_return*sol1.fun #return的是(SD_P / ER)     


    sol2 = minimize(objective_GMVP, w0, method = "SLSQP", bounds = None, constraints = CON)

    weight = sol2.x.tolist()
    GMVP_sd = sol2.fun**0.5  #return的是Var
        
    for i in range(10):
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

    answery = return_list
    answerx = sd_list
   

def solve_without_shortsale(): 

    w0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    b = (0.0, 1.0)
    bs = (b,b,b,b,b,b,b,b,b,b)

    global CON, return_list, answerx, answery

    CON = {"type":"eq","fun":constraint}

    BMVP_return = 0
    GMVP_return = 0

    sol1 = minimize(objective_BMVP, w0, method = "SLSQP", bounds = bs, constraints = CON)
        
    weight = sol1.x.tolist()
        
    for i in range(10):
        BMVP_return += weight[i]*average_return[i]

    BMVP_sd = BMVP_return*sol1.fun #return的是(SD_P / ER)     


    sol2 = minimize(objective_GMVP, w0, method = "SLSQP", bounds = bs, constraints = CON)

    weight = sol2.x.tolist()
    GMVP_sd = sol2.fun**0.5  #return的是Var
        
    for i in range(10):
        GMVP_return += weight[i]*average_return[i]
  
    distance = (BMVP_return - GMVP_return)/6

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

    answery = return_list
    answerx = sd_list      

#畫圖
def draw():

    plt.plot(anserx, answery)
    plt.savefig("/Users/xuyuxiang/Desktop/image/the_figure.png")

stock = '/Users/xuyuxiang/Desktop/raw_data.csv'
mb64 = ""
d = dict()
data_len = -1
answerx =[]
answery = []

portfolio = Portfolio(csv_to_array(stock))

portfolio.solve(method = 'M', shortsale = True)

print(name)
print(name[1])
print(name2)

root = mainpage()
root.mainloop() 
root.master.title("Investment Model") 

