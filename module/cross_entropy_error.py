# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:23:45 2019

@author: peter
"""

import numpy as np
def cross_entropy_error(y,t):
    delta =1e-7
    return -np.sum(t*np.log(y+delta))

y=np.array([0,0,1])
t=np.array([0.3,0.2,0.5])

print(cross_entropy_error(y,t))