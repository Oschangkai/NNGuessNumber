# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 11:21:07 2019

@author: peter
"""
import numpy as np
np.set_printoptions(threshold=np.inf)
import copy
def Popcount(value):
    
     val = copy.deepcopy(value)
      
     count = 0
     i = 0
     
     while val > 0:
         if(val & 0x01): #0x01 是16進位的01
             count += 1            
         val >>= 1    
         i += 1
         
     return count
     
print (Popcount(255))