import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
from time import perf_counter


def driver():

     ''' create  matrix for testing different ways of solving a square 
     linear system'''

     '''' N = size of system'''
     N = 2,5,10,25,50,75,100,500,1000,2000,4000,5000
 

     for i in range(len(N)):
          ''' Right hand side'''
          print('\n\n\nN = ',N[i])
          b = np.random.rand(N[i],1)
          A = np.random.rand(N[i],N[i])

          reg_solve_func(A,b,N[i])
          LU_solve_func(A,b,N[i])



     ''' Create an ill-conditioned rectangular matrix '''
     N = 10
     M = 5
     A = create_rect(N,M)     
     b = np.random.rand(N,1)

def LU_solve_func(A,b,n):
     tic = perf_counter() #Start time
     lu, piv = scila.lu_factor(A)
     toc = perf_counter() #end time
     print('Time for LU factorization',toc-tic)
     
     tic1 = perf_counter() #Start time
     x = scila.lu_solve((lu, piv), b)
     toc1 = perf_counter() #end time
     print('Time for LU solve',toc1-tic1)

     print('Total time for LU',toc1-tic1+toc-tic)
     
     test = np.matmul(A,x)
     r = la.norm(test-b)

     return r


def reg_solve_func(A,b,n):
     tic = perf_counter() #Start time
     x = scila.solve(A,b)
     toc = perf_counter() #end time
     print('Time for standard solve',toc-tic)
     
     test = np.matmul(A,x)
     r = la.norm(test-b)
     
     return r


def create_rect(N,M):
     ''' this subroutine creates an ill-conditioned rectangular matrix'''
     a = np.linspace(1,10,M)
     d = 10**(-a)
     
     D2 = np.zeros((N,M))
     for j in range(0,M):
        D2[j,j] = d[j]
     
     '''' create matrices needed to manufacture the low rank matrix'''
     A = np.random.rand(N,N)
     Q1, R = la.qr(A)
     test = np.matmul(Q1,R)
     A =    np.random.rand(M,M)
     Q2,R = la.qr(A)
     test = np.matmul(Q2,R)
     
     B = np.matmul(Q1,D2)
     B = np.matmul(B,Q2)
     return B     
          
  
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()       
