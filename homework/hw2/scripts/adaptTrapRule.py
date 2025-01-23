import numpy as np
import math as m
# Func to calculate each successive iteration of trapezoid rule,
# using the adaptive method derived in Lecture notes 03.5.3

def func(x):
    return ( (np.sin(np.sqrt(100.*x))) ** 2 )


def AdaptTrap(f, N, I_prev, h, a, b):
    c = I_prev / 2.
    s = 0
    for k in range(1,N):
        if k % 2 == 0:
            continue
        s += h*f(a + k*h)
    I_i = c + s
    I_err = (I_i - I_prev)/3.
    return I_i, I_prev, I_err



a = 0.
b = 1.
N = 1
eps = 1.e-06
h = 0.

Int_i = 0.
Int_im1 = 0.
err_i = 1.

# A trapezoid rule approximation with 1 slice is given by
Trap0 = (func(b) - func(a))/2.

Int_i = Trap0
# Iterate the approximation
it = 1
while np.abs(err_i) > eps:
    N = 2 ** it
    h = (b-a)/N
    Int_i, Int_im1, err_i = AdaptTrap(func, N, Int_i, h, a, b)
    print("Iteration no. "+str(it)
          +": I_"+str(it)+" = "+str(round(Int_i,7))
          +", error = "+str(round(err_i,7)))
    it += 1

print("Adaptive integration took "+str(it-1)+" iterations to converge to the provided tolerance: "+str(eps))
