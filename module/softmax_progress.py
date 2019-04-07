# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 17:24:55 2019

@author: peter
"""
import matplotlib.pylab as plt
import numpy as np
def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a -c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

x=np.array([3.0,5.0])
print(softmax(x))
