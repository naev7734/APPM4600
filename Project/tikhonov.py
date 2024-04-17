import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math

def driver():
    a = 0 #endpoints
    b = 5
    n = 20 #number of interpolation points 
    m = 7 #order of polynomial
    Lamb = .5 #Lambda
    h = (b-a)/n
    
    noise = np.random.normal(0,h,n) #create gaussian noise where 1 standard deviation is the same as the step size h

    x_int = np.linspace(a,b,n)

    f = lambda x: np.sin(x) + np.sin(5*x) #function
    f_1der = lambda x: np.cos(x) + 5*np.cos(5*x) #first derivative
    f_2der = lambda x: -1*np.sin(x) - 25*np.sin(5*x) #second derivative

    y_int = f(x_int) + noise #interpolation data


    coeff_ls = Tikhonov(x_int,y_int,m,0) #coefficients of resulting polynomial using regular least squares
    coeff_TK = Tikhonov(x_int,y_int,m,Lamb) #coefficients of resulting polynomial using tikhonov

    x_graph = np.linspace(a,b,1000) #create data for plotting smooth line of function 
    y_graph = f(x_graph)

    y_poly_ls = np.polyval(coeff_ls,x_graph) #create data for plotting smooth line of polynomial approx using regular least squares
    y_poly_TK = np.polyval(coeff_TK,x_graph) #create data for plotting smooth line of polynomial approx using tikhonov

    plt.plot(x_int,y_int,'o')
    plt.plot(x_graph,y_graph)
    plt.plot(x_graph,y_poly_ls)
    plt.plot(x_graph,y_poly_TK)
    plt.legend(['Data','True Function','Regular LS Estimate','Tikhonov Estimate'])
    plt.show()

    #DERIVATIVE STUFF
    #First derivative
    coeff_1der_ls = np.polyder(coeff_ls)
    coeff_1der_TK = np.polyder(coeff_TK)

    y_poly_1der_ls = np.polyval(coeff_1der_ls,x_graph)
    y_poly_1der_TK = np.polyval(coeff_1der_TK,x_graph)

    y_graph_1der = f_1der(x_graph)


    plt.plot(x_graph,y_graph_1der)
    plt.plot(x_graph,y_poly_1der_ls)
    plt.plot(x_graph,y_poly_1der_TK)
    plt.legend(['True Function','Regular LS Estimate','Tikhonov Estimate'])
    plt.show()

    #Second derivative
    coeff_2der_ls = np.polyder(coeff_1der_ls)
    coeff_2der_TK = np.polyder(coeff_1der_TK)

    y_poly_2der_ls = np.polyval(coeff_2der_ls,x_graph)
    y_poly_2der_TK = np.polyval(coeff_2der_TK,x_graph)

    y_graph_2der = f_2der(x_graph)


    plt.plot(x_graph,y_graph_2der)
    plt.plot(x_graph,y_poly_2der_ls)
    plt.plot(x_graph,y_poly_2der_TK)
    plt.legend(['True Function','Regular LS Estimate','Tikhonov Estimate'])
    plt.show()



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
    for i in range(m+1):
        for ii in range(n):
            A[ii,i] = x_int[ii]**i
    temp = np.matmul(np.transpose(A),A)+Lamb**2*np.identity(m+1) #Form left side of solve
    a = np.linalg.solve(temp,np.matmul(np.transpose(A),y_int)) #solve for the coefficients
    a = np.flip(a) #change the order to match np.polyval
    return a
 





if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()