import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv
from numpy.linalg import norm

def driver():

    def f(x):
        y = x**2
        return y
    
    x = np.linspace(-10,10)
    y = f(x)

    x_nodes = np.array([.1,5])
    y_nodes = f(x_nodes)
    alpha = 2
    f_alpha = line(x_nodes,y_nodes,alpha)

    plt.plot(x,y)
    plt.plot(x_nodes,y_nodes)
    plt.plot(alpha,f_alpha,'ro')
    plt.show()



def line(x,y,alpha):
    #x is vector of two nodes
    #y is vector is the function evaluated at each x
    #alpha is the interpolation node
    m = (y[1]-y[0])/(x[1]-x[0])
    y_alph = y[0]+m*(alpha-x[0])
    return[y_alph]



if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()