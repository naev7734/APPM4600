import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math

x = np.array([[0],[1],[2],[3]])
y = np.array([[1],[4],[2],[6]])

n = len(x)

M = np.array([[n+1, np.sum(x)],[np.sum(x),np.sum(x**2)]])

b = np.array([sum(y),sum(x*y)])
print(b)
a = la.solve(M,b)