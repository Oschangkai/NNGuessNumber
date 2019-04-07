# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 07:58:36 2019

@author: peter
"""
import Numpy as np

def AND(x1,x2):
    W1,W2,theta =0.5,0.5,0.7
    tmp = x1*W1+x2*W2
    if tmp <=theta:
        return 0
    if tmp >theta:
        return 1
    
def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    tmp =np.sum(w*x)
    if tmp <=0:
        return 0
    else:
        return 1
    
def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    tmp =np.sum(w*x)
    if tmp<=0:
        return 0
    else:
        return 1
    
def XOR(x1,x2):
    s1 =NAND(x1,x2)
    s2 =OR(x1,x2)
    y =AND(s1,s2)
    return y
