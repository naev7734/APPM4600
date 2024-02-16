# import libraries
import numpy as np
import matplotlib.pyplot as plt
    
# define routines
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return [xstar,ier]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]
    


# use routines 
f1 = lambda x: x-4*np.sin(2*x)-3
x_pnts = np.linspace(-5,10,10000)
y_pnts = f1(x_pnts)

plt.plot(x_pnts,y_pnts)
plt.plot([-5,10],[0,0])
plt.show()

Nmax = 100
tol = 1e-10

''' test f1 '''
x0 = 1.5
[xstar,ier] = fixedpt(f1,x0,tol,Nmax)
print('the approximate fixed point is:',xstar)
print('f1(xstar):',f1(xstar))
print('Error message reads:',ier)

f2 = lambda x: -1*np.sin(2*x)+5*x/4-.75
''' test f2 '''
x0 =-3
[xstar,ier] = fixedpt(f2,x0,tol,Nmax)
print('the approximate fixed point is:',xstar)
print('f2(xstar):',f2(xstar))
print('Error message reads:',ier)


