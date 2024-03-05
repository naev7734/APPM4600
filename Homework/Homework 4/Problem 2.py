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

    def LazyNewton(x0,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    xlist = np.zeros((Nmax+1,len(x0)));
    xlist[0] = x0;

    J = evalJ(x0);
    for its in range(Nmax):

       F = evalF(x0)
       x1 = x0 - np.linalg.solve(J,F);
       xlist[its+1]=x1;

       if (norm(x1-x0) < tol*norm(x0)):
           xstar = x1
           ier =0
           return[xstar,xlist, ier,its];

       x0 = x1

    xstar = x1
    ier = 1
    return[xstar,xlist,ier,its];