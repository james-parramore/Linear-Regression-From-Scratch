# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 22:52:19 2019

"""
import matplotlib.pyplot as plt
import pandas as pd

#import our dataset

plt.rcParams['figure.figsize'] = (12.0, 9.0)
dataset = pd.read_csv('data.csv')

# Separate X and Y values
X = dataset.iloc[:, 0]
Y = dataset.iloc[:, 1]

#plot our data
plt.scatter(X,Y)
plt.show()

#choose our learning rate and number of epochs
rate = .0001
epoch = 1000

m = 0
c = 0

#number of x values
n = float(len(X))

#perform gradient descent

for i in range(epoch):
    Yp = m*X + c      # predicted value of y
    D_m = (-2/n) * sum(X*(Y-Yp)) #derivative with respect to m
    D_c = (-2/n) * sum(Y-Yp)     #derivattive with respect to c
    m = m - rate * D_m           # update m and c
    c = c - rate * D_c

Yp = m*X + c


#plot our regression line
plt.scatter(X, Y) 
plt.plot([min(X), max(X)], [min(Yp), max(Yp)], color='red')  # regression line
plt.show()