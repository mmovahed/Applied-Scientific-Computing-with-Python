import numpy as np
import matplotlib.pyplot as plt

def func(x,y):
    return x+y


def ab2(fcn,a,b,y0,N):
    h = (b-a)/N
    x = a + np.arange (N+1) * h
    y = np.zeros(x.size)
    f = np.zeros_like(y)
    y[0] = y0
    f[0] = fcn(x[0],y[0])
    
    k1 = f[0]
    k2 = fcn( x[0] + h , y[0] + h * k1 )
    y [1] = y[0] + h*( k1 + k2 ) / 2
    f [1] = fcn(x[1],y[1])
    for k in range (1,N) :
        y [k+1] = y [k] + h * ( 3 * f[k] - f[k-1]) /2
        f [k+1] = fcn(x[k+1] , y[k+1])
    return x , y

x , y = ab2(func,1,5,1,100)
plt.plot(x, y)
plt.title('The Adams–Bashforth two-step method method RK4')
plt.xlabel('X', fontsize = 18)
plt.ylabel('Y', fontsize = 18)
plt.show()

