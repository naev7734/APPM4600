import numpy as np
from scipy import io, integrate, linalg, signal

def driver():
    n = 4
    x_eval = .5
    phi = eval_legendre(n,x_eval)
    print(phi)

def eval_legendre(n,x_eval):
    #n - order
    #x_eval - evaluation node
    phi = np.zeros(n+1)
    phi[0] = 1
    phi[1] = x_eval

    for i in range(1,n):
        phi[i+1] = (1/(i+1))*((2*i+1)*x_eval*phi[i]-i*phi[i-1])

    return(phi)

if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()