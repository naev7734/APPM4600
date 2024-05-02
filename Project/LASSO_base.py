import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
def driver():
    a = 0 #endpoints
    b = 3
    n = 40 #number of interpolation points 
    h = (b-a)/n
    num_basis = 5    

    Lamb = 1 #lambda, the coefficient for the 1-norm

    fine = 10000 #number of points to use for graphing and error calculation

    noise = np.random.normal(0,10,n) #create gaussian noise where 1 standard deviation is the same as the step size h

    x_int = np.linspace(a,b,n)

    f = lambda x: 5*x + .5*x**4 #function

    y_int = f(x_int) + noise #interpolation data

    x_graph = np.linspace(a,b,fine) #create data for plotting smooth line of function 
    y_graph = f(x_graph) #truth data

    X = basis(x_int,np.ones(num_basis),num_basis) #CHANGE FOR # OF BASIS ELEMENTS

    clf = linear_model.Lasso(Lamb) #create the LASSO model
    clf.fit(X,y_int) #train it on your x and y data
    print(clf.coef_)

    print(clf.intercept_)

    y_graph_lasso = np.sum(basis(x_graph,clf.coef_,num_basis), axis=1)

    plt.plot(x_int,y_int,'o')
    plt.plot(x_graph,y_graph)
    plt.plot(x_graph,y_graph_lasso)
    plt.legend(['Data','True Function','LASSO Approximation'])
    plt.show()


def basis(x,c,num_basis):
    X = np.zeros([len(x),num_basis])#input number of basis elements here
    for i in range(len(x)):
        X[i,0] = c[0]*1 #DEFINE BASIS HERE
        X[i,1] = c[1]*x[i]
        X[i,2] = c[2]*x[i]**2 
        X[i,3] = c[3]*x[i]**3
        X[i,4] = c[4]*x[i]**4
    return X


if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()