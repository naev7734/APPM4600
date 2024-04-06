import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math

def driver():
    f = lambda x: 1/(1+(16*x)**2)
    n = 49
    i = np.arange(0,n+1)
    x_interp = np.cos((2*i+1)*np.pi/(2*(n+1)))

    z = np.linspace(-1,1,1001)

    [p,phi] = barycentric_lagrange(f,x_interp,z)
    
    plt.plot(x_interp,f(x_interp),'o')
    plt.plot(z,p)
    plt.plot(z,f(z))
    plt.show()


def barycentric_lagrange(f,x,z):
    w = np.zeros(len(x))
    for j in range(1,len(x)):
        w[j] = 1
        for i in range(1,len(x)):
            if i != j:
                w[j] = w[j]/(x[j]-x[i])
    
    phi = np.zeros(len(z))
    p = np.zeros(len(z))
    for n in range(1,len(z)):
        phi[n] = 1
        temp = 0
        for j in range(1,len(x)):
            phi[n] = phi[n]*(z[n]-x[j])
            temp = temp + w[j]*f(x[j])/(z[n]-x[j])
        p[n] = phi[n] * temp
    return(p,phi)


if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()

