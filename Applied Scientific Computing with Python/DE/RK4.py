import numpy as np
import matplotlib.pyplot as plt

def func(x,y):
    return x+y

def RK4(fcn,a,b,y0,N):
    """
    Solve y' = f (x,y) in 'N' steps using
    fourth−order Runge−Kutta with initial
    condition y [a] = y0 .
    """
    h = (b-a) / N
    x = a + np.arange(N+1) * h
    y = np.zeros(x.size)
    y [ 0 ] = y0
    for k in range (N) :
        k1 = fcn(x[k] , y[k] )
        k2 = fcn(x[k] + h / 2 , y[k] + h * k1 / 2 )
        k3 = fcn(x[k] + h / 2 , y[k] + h * k2 / 2 )
        k4 = fcn(x[k] + h , y[k] + h * k3 )
        y[k+1] = y[k] + h * ( k1 + 2 * ( k2 + k3 ) + k4 ) / 6
    return x , y

x , y = RK4(func,1,5,1,100)
plt.plot(x, y)
plt.title('The classical Runge–Kutta fourth-order method RK4')
plt.xlabel('X', fontsize = 18)
plt.ylabel('Y', fontsize = 18)
plt.show()

