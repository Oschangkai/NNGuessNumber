# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 22:26:44 2019

@author: peter
"""

import numpy as np
def numerical_gradient(f,x):
    h=1e-4
    grad =np.zeros_like(x)#產生和x相同行裝的陣列
    
    for idx in range(x.size):
        tmp_val=x[idx]
        #計算f(x+h)
        
        x[idx]=tmp_val+h
        fxh1 =f(x)
        #計算f(x-h)
        x[idx]=tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1-fxh2)/(2*h)
        x[idx] =tmp_val

    
    return grad


    