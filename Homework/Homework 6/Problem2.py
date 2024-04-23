import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math

def driver():
    n = 600 #Number of intervals t_0,t_1,...,t_n MUST BE EVEN FOR SIMPSONS
    a = -5
    b = 5
    f = lambda s: 1/(1+s**2)
    trap = traps(a,b,f,n)
    print(trap)
 
    simp = simps(a,b,f,n)
    print(simp)

def traps(a,b,f,n):
    t = np.linspace(a,b,n+1)
    y = f(t)
    trap = 0
    for i in range(n):
        trap = trap + .5*(t[i+1]-t[i])*(y[i+1]+y[i])
    return trap

def simps(a,b,f,n):
    t = np.linspace(a,b,n+1)
    y = f(t)
    simp = 0
    for i in range(n//2):
        simp = simp + (1/6)*(t[2*i+2]-t[2*i])*(y[2*i]+4*y[2*i+1]+y[2*i+2])
    return simp










if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()