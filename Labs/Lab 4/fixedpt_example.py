# import libraries
import numpy as np
    
# define routines
def fixedpt(f,x0,tol,Nmax):
   x = x0
   ''' x0 = initial guess''' 
   ''' Nmax = max number of iterations'''
   ''' tol = stopping tolerance'''

   count = 0
   counton = 0
   while (count <Nmax):
       count = count +1
       x1 = f(x0)
       x = np.append(x,x1)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          x_new_0 = np.zeros((len(x)-1,1))
          
          for n in range(0,(len(x)-1)):
            x_new_0[n] = abs(x[n+1]-x[n])

          diff = abs(sum(x_new_0))

          while (diff > .1):
            counton = counton + 1
            print('Counton is',counton)
            print('Length of x_new_0',len(x_new_0))
            print('x0 is',x_new_0)
            x_new_1 = np.zeros((len(x_new_0)-1,1))
            for n in range(0,len(x_new_0)-1):
               x_new_1[n] = abs(x_new_0[n+1]-x_new_0[n])
            x_new_0 = x_new_1

          order = counton
          print('Order in function is',order)
          diff = abs(sum(x_new_0))
          return [xstar,ier,x,x_new_0,order]
       x0 = x1

   xstar = x1
   ier = 1
   return [xstar, ier, x]
    

# use routines 
#f1 = lambda x: x*(1+((7-x**5)/(x**2)))**3
f1 = lambda x: (10/(x+4))**.5
''' 
fixed point is alpha1 = 1.4987....
'''

#f2 = lambda x: 3+2*np.sin(x)
#''' 
#fixed point is alpha2 = 3.09... 
#'''

Nmax = 100
tol = 1e-10

''' test f1 '''
x0 = 1.5
[xstar,ier,x,x_new_0,order] = fixedpt(f1,x0,tol,Nmax)
print('the approximate fixed point is:',xstar)
print('f1(xstar):',f1(xstar))
print('Error message reads:',ier)
print('Guesses were',x)
print('Order of Convergence was', order)
    
#''' test f2 '''
#x0 = 0.0
#[xstar,ier] = fixedpt(f2,x0,tol,Nmax)
#print('the approximate fixed point is:',xstar)
#print('f2(xstar):',f2(xstar))
#print('Error message reads:',ier)
