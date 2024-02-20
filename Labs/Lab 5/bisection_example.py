# import libraries
import numpy as np

def driver():

# use routines    
    f = lambda x: np.e**(x**2+7*x-30)-1
    f_der = lambda x: (2*x+7)*np.e**(x**2+7*x-30)
    f_der2 = lambda x: (2*x+7)**2*np.e**(x**2+7*x-30)+2*np.e**(x**2+7*x-30)
    a = 2
    b = 4.5
    Nmax = 100

#    f = lambda x: np.sin(x)
#    a = 0.1
#    b = np.pi+0.1

    tol = 1e-7

    [astar,ier] = bisection(f,f_der,f_der2,a,b,tol,Nmax)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print('Condition is:',np.abs(f(astar)*f_der2(astar)/(f_der(astar)**2)))




# define routines
def bisection(f,f_der,f_der2,a,b,tol,Nmax):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      cond = np.abs(f(d)*f_der2(d)/(f_der(d)**2))
      if (cond<1):
        astar = d
        p0 = d
        for it in range(Nmax):
          p1 = p0-f(p0)/f_der(p0)
          if (abs(p1-p0)<tol):
            pstar = p1
            ier =0
            return [astar, ier]
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier]
      
driver()               


