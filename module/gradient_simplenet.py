# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 16:24:32 2019

@author: peter
"""
import sys,os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax,cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W =np.random.randn(2,3)
        
    def predict(self,x):
        return np.dot(x,self.W)
    def loss(self,x,t):
        z =self.predict(x)
        y =softmax(z)
        loss =cross_entropy_error(y,t)
        
        return loss
    
    
x = np.array([1, 2])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t) #回傳新定義函數
dW = numerical_gradient(f, net.W)




print(dW)

