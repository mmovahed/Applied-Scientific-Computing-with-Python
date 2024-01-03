import numpy as np
import matplotlib.pyplot as plt

"""
R'(t) = αR(t) − βR (t) F (t)
F' (t) = −γF (t) + δF (t) R (t)
R (0) = 500, F (0) = 200
"""
def predprey ( t , y ) :
    r = np.empty( 2 )
    r [ 0 ] = 0.7 * y [ 0 ] - 0.005 * y [ 0 ] * y [ 1 ]
    r [ 1 ] = -0.2 * y [ 1 ] + 0.001 * y [ 0 ] * y [ 1 ]
    return r

def RK4_sys(fcn,a,b,y0,N):
    """
    Solve y' = f (x,y) in 'N' steps using
    fourth−order Runge−Kutta with initial
    condition y [a] = y0.
    """
    h = (b-a) / N
    x = a + np.arange(N+1) * h
    y = np.zeros((x.size,y0.size))
    y [ 0,: ] = y0
    for k in range (N) :
        k1 = fcn(x[k] , y[k,:] )
        k2 = fcn(x[k] + h / 2 , y[k,:] + h * k1 / 2 )
        k3 = fcn(x[k] + h / 2 , y[k,:] + h * k2 / 2 )
        k4 = fcn(x[k] + h , y[k,:] + h * k3 )
        y[k+1,:] = y[k,:] + h * ( k1 + 2 * ( k2 + k3 ) + k4 ) / 6
    return x , y

RF0 = np.array([500 , 200])

x , y =RK4_sys(predprey,0,5,RF0,100)
#plt.plot(y[:,1], y[:,0])
#plt.title('The classical Runge–Kutta fourth-order method RK4')
#plt.xlabel('X', fontsize = 18)
#plt.ylabel('Y', fontsize = 18)
#plt.zlabel('Y', fontsize = 18)
#plt.show()

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot3D(x, y[:,1],  y[:,0], 'green')
ax.set_title('The classical Runge–Kutta fourth-order method RK4')
plt.show()
