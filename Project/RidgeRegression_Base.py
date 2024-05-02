import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math

def driver():
    a = 0 #endpoints
    b = 10
    n = 50 #number of interpolation points 
    m = 20 #order of polynomial
    Lamb = .3 #Lambda
    fine = 10000 #number of points to use for graphing and error calculation
    h = (b-a)/n

    noise = np.random.normal(0,h,n) #create gaussian noise where 1 standard deviation is the same as the step size h

    x_int = np.linspace(a,b,n)

    f = lambda x: np.sin(x) + np.sin(5*x) #function

    y_int = f(x_int) + noise #interpolation data


    coeff_ls = RR(x_int,y_int,m,0) #coefficients of resulting polynomial using regular least squares
    coeff_RR = RR(x_int,y_int,m,Lamb) #coefficients of resulting polynomial using tikhonov

    x_graph = np.linspace(a,b,fine) #create data for plotting smooth line of function 
    y_graph = f(x_graph)

    y_poly_ls = np.polyval(coeff_ls,x_graph) #create data for plotting smooth line of polynomial approx using regular least squares
    y_poly_RR = np.polyval(coeff_RR,x_graph) #create data for plotting smooth line of polynomial approx using tikhonov

    error_ls = np.sum(np.abs(y_graph-y_poly_ls))/fine
    error_RR = np.sum(np.abs(y_graph-y_poly_RR))/fine
    print('The average error between the true function and the polynomial using Regular Least Squares is',error_ls,'\nThe average error between the true function and the polynomial using Ridge Regression is',error_RR)

    plt.plot(x_int,y_int,'o') #Plotting data, function, LS polynomial, Tikhonov polynomial
    plt.plot(x_graph,y_graph)
    plt.plot(x_graph,y_poly_ls)
    plt.plot(x_graph,y_poly_RR)
    plt.legend(['Data','True Function','Regular LS','Ridge Regrssion'])
    plt.show()



def RR(x_int,y_int,m,Lamb):
    #Inputs:
    #x_int - interpolation nodes
    #y_int - interpolation data
    #m - degree of polynomial
    #z - x points to be estimated
    #Lamb - weight factor lambda
    #Outputs: a - coefficients for polynomial in order a_m, a_m-1, ... , a_0
    n = len(x_int) #number of interpolation points
    A = np.zeros([n,m+1])

    for i in range(m+1):
        for ii in range(n):
            A[ii,i] = x_int[ii]**i #Vandemonde matrix

    temp = np.matmul(np.transpose(A),A)+Lamb**2*np.identity(m+1) #Form left side of solve
    a = np.linalg.solve(temp,np.matmul(np.transpose(A),y_int)) #solve for the coefficients
    a = np.flip(a) #change the order to match np.polyval
    return a
 
if __name__ == '__main__':
  # run the drivers only if this is called from the command line
   driver()