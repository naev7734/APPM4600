#import os
#%system('cls')
import numpy as np
import matplotlib.pyplot as plt
import math as m

## Problem 1
#Define x
x = np.arange(1.920,2.080,.001)

#Find p via coefficients
p_coeff = x**9 - 18*x**8 + 144*x**7 - 672*x**6 + 2016*x**5 - 4032*x**4 + 5376*x**3 - 4608*x**2 + 2304*x - 512

#Find p via factored expression
p_fact = (x-2)**9

plt.plot(x,p_coeff, label='Coefficient p')
plt.plot(x,p_fact, label='Factored p')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()
plt.show()

## Problem 5 b
#Define x1
x = np.pi

delta = np.arange(0,16,1)
delta = (1/10)**delta

y_subtract = np.cos(x+delta)-np.cos(x)
y_nosubtract = -2*np.sin(x*.5*delta)*np.sin(.5*delta)
y_diff_pi = y_nosubtract - y_subtract

x = 10**6
y_subtract = np.cos(x+delta)-np.cos(x)
y_nosubtract = -2*np.sin(x*.5*delta)*np.sin(.5*delta)
y_diff_big = y_nosubtract - y_subtract

plt.semilogx(delta,y_diff_pi, label='x=pi')
plt.semilogx(delta,y_diff_big, label='x=10^6')
plt.xlabel('delta')
plt.ylabel('Difference in y')
plt.legend()
plt.show()

## Problem 5 c
x = np.pi

delta = np.arange(0,16,1)
delta = (1/10)**delta

y_subtract = np.cos(x+delta)-np.cos(x)
y_new = -1*delta*np.sin(x)-((delta**2)/2)*np.cos(x+.5*delta)
y_diff_pi = y_nosubtract - y_subtract

x = 10**6
y_subtract = np.cos(x+delta)-np.cos(x)
y_new = -1*delta*np.sin(x)-((delta**2)/2)*np.cos(x+.5*delta)
y_diff_big = y_nosubtract - y_subtract

plt.semilogx(delta,y_diff_pi, label='x=pi')
plt.semilogx(delta,y_diff_big, label='x=10^6')
plt.xlabel('delta')
plt.ylabel('Difference in y')
plt.legend()
plt.show()

