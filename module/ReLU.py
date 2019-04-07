# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:32:06 2019

@author: peter
"""
import numpy as np
import matplotlib.pylab as plt
def ReLU(X):
    return np.maximum(0,x)

x=np.arange(-5.0,5.0,1)
y=ReLU(x)
plt.plot(x,y)
plt.ylim(-0.1,10.0)
plt.show()
