import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
def driver():
    'Inputs'
    a = 0 #endpoints
    b = 5
    n = 20 #number of interpolation points
    h = (b-a)/n
    num_basis = 31    

    Lamb_1 = .1 #lambda, the coefficient for the 1-norm
    Lamb_2 = 1 #lambda for the 2-norm

    fine = 10000 #number of points to use for graphing and error calculation

    'Setup'
    noise = np.random.normal(0,h,n) #create gaussian noise where 1 standard deviation is the same as the step size h

    x_int = np.linspace(a,b,n)

    f = lambda x: np.sin(x) + np.sin(5*x) #function

    y_int = f(x_int) + noise #interpolation data

    x_graph = np.linspace(a,b,fine) #create data for plotting smooth line of function 
    y_graph = f(x_graph) #truth data

    'LASSO'
    X = basis(x_int,np.ones(num_basis),num_basis) #CHANGE FOR # OF BASIS ELEMENTS

    clf = linear_model.Lasso(Lamb_1) #create the LASSO model
    clf.fit(X,y_int) #train it on your x and y data
    print(clf.coef_)

    print(clf.intercept_)

    y_graph_lasso = np.sum(basis(x_graph,clf.coef_,num_basis), axis=1)

    'Regular Least Squares and Ridge Regression'
    coeff_ls = RR(x_int,y_int,num_basis,0) #coefficients of resulting polynomial using regular least squares
    coeff_RR = RR(x_int,y_int,num_basis,Lamb_2) #coefficients of resulting polynomial using RR    
    y_graph_ls = np.sum(basis(x_graph,coeff_ls,num_basis), axis=1) #create data for plotting smooth line of polynomial approx using regular least squares
    y_graph_RR = np.sum(basis(x_graph,coeff_RR,num_basis), axis=1) #create data for plotting smooth line of polynomial approx using RR 
    print(coeff_RR)

    'Plot Function and Approximation'
    plt.plot(x_int,y_int,'o')
    plt.plot(x_graph,y_graph)
    #plt.plot(x_graph,y_graph_ls)
    plt.plot(x_graph,y_graph_RR)
    plt.plot(x_graph,y_graph_lasso)
    #plt.legend(['Data','True Function','Regular LS','Ridge Regression','LASSO Approximation']) 
    plt.show()

    'Error'
    error_ls = np.sum(np.abs(y_graph-y_graph_ls))/fine
    error_RR = np.sum(np.abs(y_graph-y_graph_RR))/fine
    error_LASSO = np.sum(np.abs(y_graph-y_graph_lasso))/fine
    print('The average error between the true function and the approximation using Regular Least Squares is',error_ls,'\nThe average error between the true function and the approximation using Ridge Regression is',error_RR,'\nThe average error between the true function and the approximation using LASSO is',error_LASSO)

    plt.semilogy(x_graph,abs(y_graph-y_graph_ls)) #Plotting error
    plt.semilogy(x_graph,abs(y_graph-y_graph_RR))
    plt.semilogy(x_graph,abs(y_graph-y_graph_lasso))    
    plt.legend(['Regular LS Error','Ridge Regression Error','LASSO Error'])
    plt.show()



def basis(x,c,num_basis):
    X = np.zeros([len(x),num_basis])#input number of basis elements here
    for i in range(len(x)):
        X[i,0] = c[0]*1 #DEFINE BASIS HERE
        X[i,1] = c[1]*np.sin(x[i])
        X[i,2] = c[2]*np.cos(x[i])
        X[i,3] = c[3]*np.sin(2*x[i])
        X[i,4] = c[4]*np.cos(2*x[i])
        X[i,5] = c[5]*np.sin(3*x[i])
        X[i,6] = c[6]*np.cos(3*x[i])
        X[i,7] = c[7]*np.sin(4*x[i])
        X[i,8] = c[8]*np.cos(4*x[i])
        X[i,9] = c[9]*np.sin(5*x[i])
        X[i,10] = c[10]*np.cos(5*x[i])
        X[i,11] = c[11]*np.sin(6*x[i])
        X[i,12] = c[12]*np.cos(6*x[i])
        X[i,13] = c[13]*np.sin(7*x[i])
        X[i,14] = c[14]*np.cos(7*x[i])
        X[i,15] = c[15]*np.sin(8*x[i])
        X[i,16] = c[16]*np.cos(8*x[i])

        X[i,17] = c[17]*x[i]
        X[i,18] = c[18]*x[i]**2
        X[i,19] = c[19]*x[i]**3
        X[i,20] = c[20]*x[i]**4
        X[i,21] = c[21]*x[i]**5
        X[i,22] = c[22]*x[i]**6
        X[i,23] = c[23]*x[i]**7
        X[i,24] = c[24]*x[i]**8

        X[i,25] = c[25]*np.exp(.5*x[i])
        X[i,26] = c[26]*np.exp(x[i])
        X[i,27] = c[27]*np.exp(2*x[i])
        X[i,28] = c[28]*np.exp(3*x[i])
        X[i,29] = c[29]*np.exp(4*x[i])
        X[i,30] = c[30]*np.exp(5*x[i])
    return X

def RR(x_int,y_int,num_basis,Lamb):
    #Inputs:
    #x_int - interpolation nodes
    #y_int - interpolation data
    #m - degree of polynomial
    #z - x points to be estimated
    #Lamb - weight factor lambda
    #Outputs: a - coefficients for polynomial in order a_m, a_m-1, ... , a_0
    n = len(x_int) #number of interpolation points

    A = basis(x_int,np.ones(num_basis),num_basis)

    a = np.linalg.inv(A.T @ A + Lamb * np.eye(num_basis)) @ A.T @ y_int#solve for the coefficients
    return a


if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()