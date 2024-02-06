import numpy as np 
import math as m
import matplotlib.pyplot as plt

## Problem 3
x = 9.999999995000000*10**(-10)
#y = m.e**x
#print(y-1)

y = x + x**2/m.factorial(2)
print(y)

## Problem 4
#Part a
t = np.arange(0,np.pi,np.pi/30)

y = np.cos(t)

S = 0

for k in range(0,len(t)):
    S = S + t[k]*y[k]

print("the sum is:",S)

#Part b
theta = np.linspace(0,2*np.pi,100)
R = 1.2
deltar = .1
f = 15
p = 0
x = R*(1+deltar*np.sin(f*theta+p))*np.cos(theta)
y = R*(1+deltar*np.sin(f*theta+p))*np.sin(theta)

plt.plot(theta,x)
plt.plot(theta,y)
plt.show()

deltar = .05
for i in range(0,9):
    R = i
    f = 2+i
    p = np.random.uniform(0,2)
    x = R*(1+deltar*np.sin(f*theta+p))*np.cos(theta)
    y = R*(1+deltar*np.sin(f*theta+p))*np.sin(theta)
    plt.plot(theta,x)
    plt.plot(theta,y)
plt.show()