# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 22:01:49 2019

@author: peter
"""


def numerical_diff(f,x):#微分
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)