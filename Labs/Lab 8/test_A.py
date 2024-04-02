import numpy as np
N = 5

A = np.zeros((N+1,N+1))
A[0,0] = 1
A[N,N] = 1
for i in range(1,N):
    A[i,i] = 4
    if i-1 >= 0:
        A[i,(i-1)] = 1
    if i+1 <= N+1:
        A[i,i+1] = 1
        
print(A)