# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:55:46 2019

@author: peter
"""
import numpy as np
#第一層
x1=np.array([1.0,0.5])
w1 =np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
b1=np.array([0.1,0.2,0.3])

A1 =np.dot(x1,w1)+b1
print (A1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

z1=sigmoid(A1)
print (z1)

#第二層
w2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
b2=np.array([0.1,0.2])

A2=np.dot(z1,w2)+b2
z2=sigmoid(A2)
print(z2)
#第三層
def identity_function(x):#恆等函數
    return x

w3=np.array([[0.1,0.3],[0.2,0.4]])
b3=np.array([0.2,0.2])

A3=np.dot(z2,w3)+b3
y=identity_function(A3)

