# Find an optimal window by matching its frequency response with ideal
import numpy as np
import cvxopt
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

# Window length
N=512

# Oversampling (for plotting)
os=4

# "Half" Fourier basis (cosine part only)
n=np.arange(N)
k=np.arange(0,N)
F=np.cos(-2*np.pi*np.outer(k,n)/N)

# Number of cosines to sum
M=4
m=np.arange(M)
# Cosines to sum to get window
C=np.cos(2*np.pi*np.outer(n,m)/N)*np.power(-1.,np.arange(M))

# P matrix for optimization problem
P=block_diag(F,np.zeros(C.shape))
P=np.dot(P.T,P)

# q vector for optimization problem
q=np.zeros((N+M,1))
q[0]=1.
q=-2.*np.dot(P.T,q)

# A constraint matrix for optimization problem
A=block_diag(-1*np.identity(F.shape[1]),np.ones((1,M)))
A[:C.shape[0],F.shape[1]:]=C
# constrain x[0] to be 0
a_=np.zeros((1,M+N))
#a_[0]=1.
a_[0,N:]=C[0,:]
A=np.concatenate((A,a_),axis=0)

# b vector for optimization problem
b=np.zeros((C.shape[0]+1,1))
b[-1]=1.
# constrain x[0] to be 0
b=np.concatenate((b,[[0.]]),axis=0)

# G inequality matrix
G=np.concatenate((np.identity(F.shape[1]),np.zeros((F.shape[1],C.shape[1]))),axis=1)
G=np.concatenate((G,-G),axis=0)

# h inequality vector
h=np.concatenate((np.ones(F.shape[1]),np.zeros(F.shape[1])),axis=0)

P=cvxopt.matrix(P)
q=cvxopt.matrix(q)
A=cvxopt.matrix(A)
b=cvxopt.matrix(b)
G=cvxopt.matrix(G)
h=cvxopt.matrix(h)

sol=cvxopt.solvers.qp(P,q,G=G,h=h,A=A,b=b)

if sol['status'] == 'optimal':
    print 'solution found'
    print 'x='
    print sol['x'][:N]
    print 'F*x='
    print np.dot(F,np.array(sol['x'][:N]))
    print 'C*a='
    print np.dot(C,np.array(sol['x'][N:]))
    print 'a='
    print sol['x'][N:]
    fig, (ax1,ax2) = plt.subplots(2,1)
    ax1.plot(np.arange(N),np.array(sol['x'][:N]))
    f=np.fft.fft(np.array(sol['x'][:N]).flatten(),os*N)
    f_max=np.max(np.abs(f))
    ax2.plot(np.arange(0,N,1./os),20*np.log10(np.abs(f))-20*np.log10(f_max))
    plt.show()
else:
    print 'no solution found'
    print 'status: %s' % (sol['status'],)
