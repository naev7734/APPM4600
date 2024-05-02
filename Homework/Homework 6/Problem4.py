import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad
from scipy import special

def driver():
    t = [2,4,6,8,10]
    a = 0
    b = 100 #change so adequately close to inf
    n = 1000000

    gam_trap = np.zeros(len(t))
    gam_spec = np.zeros(len(t))
    gam_exact = np.zeros(len(t))
    gam_quad = np.zeros(len(t))
    gam_lagg = np.zeros(len(t))

    for i in range(len(t)):
        G = lambda x: x**(t[i]-1)*np.exp(-1*x)
        gam_trap[i] = traps(a,b,G,n)
        gam_spec[i] = special.gamma(t[i])
        gam_exact[i] = math.factorial(t[i]-1)

        #part b
        gam_quad[i], temp, dic = quad(G,a,b,full_output=1)
        
        #part c
        w,x = np.polynomial.laguerre.laggauss(100)
        for j in range(len(x)):
            gam_lagg[i] = gam_lagg[i] + w[j]*x[j]**(t[i]-1)

#gam_lagg[i] = gam_lagg[i] + .5*(x[j+1]-x[j])*(y[j+1]+y[j])        
#gam_lagg[i] = gam_lagg[i] + w[j]*G(x[j])
    print('Trapezodial Approximation:',gam_trap)
    print('Average Error Using Trapezoidal:',np.linalg.norm(np.abs(gam_trap-gam_exact)))
    
    print('Built in Function',gam_spec)
    print('Average Error using Built in Function:',np.linalg.norm(np.abs(gam_spec-gam_exact)))

    print('Quad',gam_quad)
    print('Average Error using quad:',np.linalg.norm(np.abs(gam_quad-gam_exact)))

    print('Laguerre Laggauss',gam_lagg)
    print('Average Error using Laguerre Laggauss:',np.linalg.norm(np.abs(gam_lagg-gam_exact)))


def traps(a,b,f,n):
    t = np.linspace(a,b,n+1)
    y = f(t)
    trap = 0
    for i in range(n):
        trap = trap + .5*(t[i+1]-t[i])*(y[i+1]+y[i])
    return trap

if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()