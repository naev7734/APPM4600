import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math

def driver():
    f = lambda x: 1/(1+(16*x)**2)
    n = 49
    z = np.linspace(-1,1,1001)

    i = np.arange(1,n+2)
    x_interp = -1 + (i-1)*(2/n)

    [p,psi_equi] = barycentric_lagrange(f,x_interp,z)
    
    plt.plot(x_interp,f(x_interp),'o')
    plt.plot(z,p)
    plt.plot(z,f(z))
    plt.show()


    i = np.arange(0,n+1)
    x_interp = np.cos((2*i+1)*np.pi/(2*(n+1)))

    [p,psi_cheb] = barycentric_lagrange(f,x_interp,z)
    
    plt.plot(x_interp,f(x_interp),'o')
    plt.plot(z,p)
    plt.plot(z,f(z))
    plt.show()

    plt.semilogy(z,psi_equi,label='Equispaced Nodes')
    plt.semilogy(z,psi_cheb,label='Chebyshev Nodes')
    plt.legend()
    plt.show()


def barycentric_lagrange(f,x,z):
    w = np.zeros(len(x))
    for j in range(1,len(x)):
        w[j] = 1
        for i in range(1,len(x)):
            if i != j:
                w[j] = w[j]/(x[j]-x[i])
    
    psi = np.zeros(len(z))
    p = np.zeros(len(z))
    for n in range(1,len(z)):
        psi[n] = 1
        temp = 0
        for j in range(1,len(x)):
            psi[n] = psi[n]*(z[n]-x[j])
            temp = temp + w[j]*f(x[j])/(z[n]-x[j])
        p[n] = psi[n] * temp
    print(psi)
    return(p,psi)


if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()

