# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 20:08:36 2019

@author: peter
"""
from 乘法層multipleLayer import MulLayer 
apple =100
apple_num =2
tax =1.1

# layer
mul_apple_layer =MulLayer()
mul_tax_layer =MulLayer()

#forward
apple_price =mul_apple_layer.forward(apple, apple_num)
price =mul_tax_layer.forward(apple_price, tax)

print(price)

#backward
dprice =1
dapple_price,dtax =mul_tax_layer.backward(dprice)
dapple,dapple_num =mul_tax_layer.backward(dapple_price)

print("price:", int(price))
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))
print("dTax:", dtax)