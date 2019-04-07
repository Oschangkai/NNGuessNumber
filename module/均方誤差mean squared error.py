# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:17:05 2019

@author: peter
"""
import numpy as np
def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

