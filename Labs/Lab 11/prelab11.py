import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math

def driver():
  f = lambda x: x

  a = 0
  b = 1
  N = 102
  grel_trap = trap(a,b,f,N)
  print(grel_trap)
  grel_simp = simp(a,b,f,N)
  print(grel_simp)


def trap(a,b,f,N):
  x = np.linspace(a,b,N)
  h = (b-a)/(N-1)
  y = f(x)
  grel = 0
  for i in range(N-1):
    grel = grel + .5*(y[i]+y[i+1])*h
  return(grel)

def simp(a,b,f,N):
  x = np.linspace(a,b,N)
  y = f(x)
  h = (b-a)/(N-1)
  grel = 0
  for i in range(N-1):
    grel = grel + (h/6) * (y[i]+4*f(.5*(x[i]+x[i+1]))+y[i+1])
  return(grel)

if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()