import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
def driver():
    'Inputs'
    a = 0 #endpoints
    b = 3
    n = 40 #number of interpolation points 
    h = (b-a)/n
    alpha = 1 #amount of regularization, coefficient for both 1 and 2 norm
    l1_ratio = .5 #ratio of 1 and 2 norm
    fine = 10000 #number of points to use for graphing and error calculation
    num_basis = 5
    
    f = lambda x: 5*x**3#function

    'Setup'
    noise = np.random.normal(0,30,n) #create gaussian noise where 1 standard deviation is the same as the step size h

    x_int = np.linspace(a,b,n)
    y_int = f(x_int) + noise #interpolation data

    x_graph = np.linspace(a,b,fine) #create data for plotting smooth line of function 
    y_graph = f(x_graph) #truth data

    X = basis(x_int,np.ones(num_basis),num_basis) #CHANGE FOR # OF BASIS ELEMENTS

    clf = linear_model.ElasticNet(alpha=alpha,l1_ratio=l1_ratio) #create the elastic net model
    clf.fit(X,y_int) #train it on your x and y data
    print(clf.coef_)

    print(clf.intercept_)

    y_graph_en = np.sum(basis(x_graph,clf.coef_,num_basis), axis=1)

    'Graphing'
    plt.plot(x_int,y_int,'o')
    plt.plot(x_graph,y_graph)
    plt.plot(x_graph,y_graph_en)
    plt.legend(['Data','True Function','Elastic Net'])
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

def Tikhonov(x_int,y_int,m,Lamb):
    #Inputs:
    #x_int - interpolation nodes
    #y_int - interpolation data
    #m - degree of polynomial
    #z - x points to be estimated
    #Lamb - weight factor lambda
    #Outputs: a - coefficients for polynomial in order a_m, a_m-1, ... , a_0
    n = len(x_int) #number of interpolation points
    A = np.zeros([n,m+1])
    D = np.zeros([m-1,m+1])
    for i in range(m-1):
        D[i,i] = -0.5
        D[i,i+2] = 0.5

    for i in range(m+1):
        for ii in range(n):
            A[ii,i] = x_int[ii]**i
    temp = np.matmul(np.transpose(A),A)+Lamb**2*np.matmul(np.transpose(D),D) #Form left side of solve
    a = np.linalg.solve(temp,np.matmul(np.transpose(A),y_int)) #solve for the coefficients
    a = np.flip(a) #change the order to match np.polyval
    return a

if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()