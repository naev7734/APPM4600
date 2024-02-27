import numpy as np
import matplotlib.pyplot as plt

def fordiff(f,s,h):
    f_prime_s = (f(s+h)-f(s))/h
    return[f_prime_s]

def centerdiff(f,s,h):
    f_prime_s = (f(s+h)-f(s-h))/(2*h)
    return[f_prime_s]

f = lambda x: np.cos(x)
x = np.pi/2
x_prime = np.array(-1)
h = 0.01*2.**(-np.arange(0, 10))

f_prime_fordiff = np.array(fordiff(f,x,h))
print('F\' computed with a forward difference:',f_prime_fordiff)
e_fordiff = np.array(np.abs(x_prime-f_prime_fordiff))
e_fordiff = np.transpose(e_fordiff)

f_prime_centerdiff = centerdiff(f,x,h)
print('\nF\' computed with a center difference:',f_prime_centerdiff)
e_centerdiff = np.array(np.abs(x_prime-f_prime_centerdiff))
e_centerdiff = np.transpose(e_centerdiff)

plt.plot(-1*np.log10(h),np.log10(e_fordiff))
plt.plot(-1*np.log10(h),np.log10(e_centerdiff))

print('\n Both approximations are first order and have linear convergence.')

plt.show()