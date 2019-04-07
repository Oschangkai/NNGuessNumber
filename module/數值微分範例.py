# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 22:05:12 2019

@author: peter
"""
import numpy as np
import matplotlib.pylab as plt

def function_1(x):#二次函數
    return 0.01*x**2+0.1*x

def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h)/(2*h))
    
x=np.arange(0.0,20.0,0.1)
y=function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show
