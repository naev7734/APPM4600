import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv
from numpy.linalg import norm

def evalF(x):

    F = np.zeros(2)

    F[0] = 3*x[0]**2-x[1]**2
    F[1] = 3*x[0]*x[1]**2-x[0]**3-1
    return F

def evalJ(x):


    J = np.array([[6*x[0],3*x[1]**2-3*x[0]**2],[-2*x[1],6*x[0]*x[1]]])
    return J


def Newton(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    xlist = np.zeros((Nmax+1,len(x0)));
    xlist[0] = x0;

    for its in range(Nmax):
       J = evalJ(x0);
       F = evalF(x0);

       x1 = x0 - np.linalg.solve(J,F);
       xlist[its+1]=x1;

       if (norm(x1-x0) < tol*norm(x0)):
           xstar = x1
           ier =0
           return[xstar, xlist,ier, its];

       x0 = x1

    xstar = x1
    ier = 1
    return[xstar,xlist,ier,its];

# PART A
f = lambda x,y: 3*x**2-y**2
g = lambda x,y: 3*x*y**2-x**3-1

x_0 = 1
y_0 = 1
tol = np.array(1e-10)
Nmax = 100

A_mat = np.array([[1/6,1/18],[0,1/6]])

x_mat = np.array([[x_0],[y_0]])

f_xy = f(x_mat[0],x_mat[1])
g_xy = g(x_mat[0],x_mat[1])

count = 0
while count <Nmax and np.abs(f_xy) >tol and np.abs(g_xy) >0:
    f_xy = f(x_mat[0],x_mat[1])
    g_xy = g(x_mat[0],x_mat[1])
    f_eval = np.array([f_xy,g_xy])
    x_mat = x_mat-np.matmul(A_mat,f_eval)
    count = count + 1

f_xy = f(x_mat[0],x_mat[1])
g_xy = g(x_mat[0],x_mat[1])
f_eval = np.array([f_xy,g_xy])

print('PART A')
print('Solution at:',x_mat)
print('Took',count,'iterations')
print('Remainder is',f_eval,'\n\n')


# PART C
x_0 = 1
y_0 = 1
tol = np.array(1e-10)
Nmax = 10000
x0 = np.array([x_0,y_0])
[xstar,xlist,ier,its] =  Newton(x0,tol,Nmax)

print('PART C')
print('Solution at:',xstar)
print('Took',its,'iterations')