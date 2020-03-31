"""
hw6.py
Name(s): Aidan Fitzsimons
NetId(s): azf6
Date: April 3, 2020
"""

import math
import numpy as np
import matplotlib.pyplot as plt

"""
FE: Forward Euler
"""
def FE(w0, z, m, w, x0, T, N):
    A = np.matrix([[0,1],[(-1*(w0**2)),(-2*w0*z)]])
    dt = T/N
    xvec = [[0 for i in range(N+1)] for j in range(2)]
    t = list(range(N+1))
    x = [0] * (N+1)
    for i in range(N+1):
        t[i] = t[i] * dt
    xvec[0][0] = x0[0]
    xvec[1][0] = x0[1]
    for ii in range(N):
        xn = np.matrix([[float(xvec[0][ii])],[float(xvec[1][ii])]])
        bn = ([[0],[math.cos(w*t[ii])/m]])
        xn1 = (xn + np.multiply(dt, np.matmul(A,xn)) + np.multiply(dt,bn))
        xvec[0][ii+1] = float(xn1[0]) 
        xvec[1][ii+1] = float(xn1[1])
    for ii in range(len(xvec[0])):
        x[ii] = float(xvec[0][ii])
    return (x,t)

"""
BE: Backward Euler
"""
def BE(w0, z, m, w, x0, T, N):
    A = np.matrix([[0,1],[(-1*(w0**2)),(-2*w0*z)]])
    dt = T/N
    tA = np.multiply(dt,A)
    I = np.matrix([[1,0],[0,1]])
    ItA = np.linalg.inv((I - tA))
    xvec = [[0 for i in range(N+1)] for j in range(2)]
    t = list(range(N+1))
    x = [0] * (N+1)
    for i in range(N+1):
        t[i] = t[i] * dt
    xvec[0][0] = x0[0]
    xvec[1][0] = x0[1]
    for ii in range(N):
        xn = np.matrix([[float(xvec[0][ii])],[float(xvec[1][ii])]])
        bn1 = ([[0],[math.cos(w*t[ii+1])/m]])
        addStep = xn + np.multiply(dt,bn1)
        xn1 = np.matmul(ItA,addStep)
        xvec[0][ii+1] = (xn1[0]) 
        xvec[1][ii+1] = (xn1[1])
    for ii in range(len(xvec[0])):
        x[ii] = float(xvec[0][ii])
    return (x,t)

"""
CN: Crank-Nicolson
"""
def CN(w0, z, m, w, x0, T, N):
    A = np.matrix([[0,1],[(-1*(w0**2)),(-2*w0*z)]])
    dt = T/N
    tA = np.multiply((dt/2),A)
    I = np.matrix([[1,0],[0,1]])
    ItA = np.linalg.inv((I - tA))
    xvec = [[0 for i in range(N+1)] for j in range(2)]
    t = list(range(N+1))
    x = [0] * (N+1)
    for i in range(N+1):
        t[i] = t[i] * dt
    xvec[0][0] = x0[0]
    xvec[1][0] = x0[1]
    for ii in range(N):
        xn = np.matrix([[float(xvec[0][ii])],[float(xvec[1][ii])]])
        bn1 = ([[0],[math.cos(w*t[ii+1])/m]])
        bn = ([[0],[math.cos(w*t[ii])/m]])
        bterm = (1/2)*(np.add(bn,bn1))
        It2A = I + np.multiply((dt/2),A)
        addStep = np.matmul(It2A,xn) + np.multiply(dt,bterm)
        xn1 = np.matmul(ItA,addStep)
        xvec[0][ii+1] = (xn1[0]) 
        xvec[1][ii+1] = (xn1[1])
    for ii in range(len(xvec[0])):
        x[ii] = float(xvec[0][ii])
    return (x,t)

"""
RK4: fourth order Runge-Kutta
"""
def RK4(w0, z, m, w, x0, T, N):
    A = np.matrix([[0,1],[(-1*(w0**2)),(-2*w0*z)]])
    dt = T/N
    xvec = [[0 for i in range(N+1)] for j in range(2)]
    t = list(range(N+1))
    x = [0] * (N+1)
    for i in range(N+1):
        t[i] = t[i] * dt
    xvec[0][0] = x0[0]
    xvec[1][0] = x0[1]
    for ii in range(N):
        xn = np.matrix([[float(xvec[0][ii])],[float(xvec[1][ii])]])
        bn = ([[0],[math.cos(w*t[ii])/m]])
        k1 = (xn + np.multiply(dt, np.matmul(A,xn)) + np.multiply(dt,bn))

        #k2 add step
        addstep = xn + (k1/2)
        k2 = (addstep + np.multiply(dt/2, np.matmul(A,addstep))
              + np.multiply(dt/2,bn))

        #k3 add step
        addstep = xn + (k2/2)
        k3 = (addstep + np.multiply(dt/2, np.matmul(A,addstep))
              + np.multiply(dt/2,bn))

        #k4 addstep
        addstep = xn + k3
        k4 = (addstep + np.multiply(dt, np.matmul(A,addstep))
              + np.multiply(dt,bn))

        xn1 = xn + (1/6)*(k1 + k2 + k2 + k3 + k3 + k4)
        
        xvec[0][ii+1] = float(xn1[0]) 
        xvec[1][ii+1] = float(xn1[1])
    for ii in range(len(xvec[0])):
        x[ii] = float(xvec[0][ii])
    return (x,t)

"""
main
"""
if __name__ == '__main__':
    w0 = 1
    z = 1
    m = 1
    w = 1
    x0 = np.matrix([[0],[0]])
    T = 10
    N = [100, 1000, 10000, 100000, 100000]
    FElast = [0] * 5
    BElast = [0] * 5
    CNlast = [0] * 5
    RKlast = [0] * 5
    answer = 0.5*(math.sin(10)-(10*math.exp(-10)))
    print(answer)

    for i in range(len(N)):
        n = N[i]
        [x,t] = FE(w0,z,m,w,x0,T,n)
        FElast[i] = abs(x[len(x)-1] - answer)
        [x,t] = BE(w0,z,m,w,x0,T,n)
        BElast[i] = abs(x[len(x)-1] - answer)
        [x,t] = CN(w0,z,m,w,x0,T,n)
        CNlast[i] = abs(x[len(x)-1] - answer)
    print(FElast)
    print(BElast)
    print(CNlast)

    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(np.log10(N), np.log10(FElast), label = 'Error')
    legend = ax.legend(loc = 'upper left')
    plt.title('Error using FE estimate')
    plt.xlabel('N (log base 10)')
    plt.ylabel('Error (log base 10)')
    plt.savefig('FE.png',bbox_inches = 'tight')
    plt.close()

    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(np.log10(N), np.log10(BElast), label = 'Error')
    legend = ax.legend(loc = 'upper left')
    plt.title('Error using BE estimate')
    plt.xlabel('N (log base 10)')
    plt.ylabel('Error (log base 10)')
    plt.savefig('BE.png',bbox_inches = 'tight')
    plt.close()

    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(np.log10(N), np.log10(CNlast), label = 'Error')
    legend = ax.legend(loc = 'upper left')
    plt.title('Error using CN estimate')
    plt.xlabel('N (log base 10)')
    plt.ylabel('Error (log base 10)')
    plt.savefig('CN.png',bbox_inches = 'tight')
    plt.close()
                    
    
