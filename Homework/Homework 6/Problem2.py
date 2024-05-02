import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad

def driver():
    #Part a
    n = 20 #Number of intervals t_0,t_1,...,t_n MUST BE EVEN FOR SIMPSONS
    a = -5
    b = 5
    f = lambda s: 1/(1+s**2)
    trap = traps(a,b,f,n)
    print(trap)
 
    simp = simps(a,b,f,n)
    print(simp)

    #Part b
    n_trap = 1291
    n_simp = 108

    intexact = 2*np.arctan(5)

    trap = traps(a,b,f,n_trap)
    print('\nPart b:\nExact error using trapezoidal is',np.abs(trap-intexact) )

    simp = simps(a,b,f,n_simp)
    print('Exact error using Simpsons is',np.abs(simp-intexact) )

    #Part c
    trap_quad = quad(f,a,b,limit=n_trap)
    print('\nPart c:\n Error using quad with n_trap=1291 points:',np.abs(trap_quad[0]-intexact))

    simp_quad = quad(f,a,b,limit=n_simp)
    print('Error using quad with n_simp=108 points:',np.abs(simp_quad[0]-intexact))


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