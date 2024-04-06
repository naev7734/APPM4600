import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
from scipy.interpolate import CubicSpline

f = lambda x: np.sin(9*x)

x_graph = np.linspace(0,1,1000)
y_graph = f(x_graph)

n = [5,10,20,40]

for i in range(len(n)):
    x = np.linspace(0,1,n[i])
    y_exact = f(x)
    cs = CubicSpline(x,y_exact)
    y_interp = cs(x_graph)
    plt.semilogy(x_graph,np.abs(y_graph-y_interp),label=n[i])





#plt.plot(x_graph,y_graph)
#plt.plot(x_graph,y_interp)
#plt.show()

plt.legend()
plt.show()



#for i in range(len(n)):

