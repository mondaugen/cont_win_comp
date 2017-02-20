# Find an optimal window by matching its frequency response with ideal
# This matches the mid-points of the lobes of the DFT rather than the troughs
# Because this uses a least-squared error, the solution is not the same as the
# Blackman-Harris solution (the minimum height of the highest side lobe). To get
# that, you must use the L-infinity norm.
import numpy as np
import cvxopt
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

# Window length (non-zero part)
N=512

# Oversampling (for plotting)
os=4

# L is total length of window including zero-padding
L=2*N

# "Half" Fourier basis (cosine part only)
n=np.arange(N)
l=np.arange(L)
k=np.arange(1,L,2)
# First bin centred at frequency 0 (DC)
k=np.concatenate(([0],k))
F_=np.cos(-2*np.pi*np.outer(k,l)/L)
F=F_[:,((L-N)/2):(-(L-N)/2)]

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
#h=np.concatenate((np.ones(F.shape[1]),np.zeros(F.shape[1])),axis=0)
h=np.concatenate((np.ones(F.shape[1]),1*np.ones(F.shape[1])),axis=0)

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
    fig, (ax1,ax2,ax3) = plt.subplots(3,1)
    ax1.plot(np.arange(N),np.array(sol['x'][:N]))
    ax1.set_title('window')
    f=np.fft.fft(np.array(sol['x'][:N]).flatten(),os*N)
    f_max=np.max(np.abs(f))
    ax2.plot(np.arange(0,N,1./os),20*np.log10(np.abs(f))-20*np.log10(f_max))
    ax2.set_title('DFT of window %d x oversampled' % (os,))
    f_opt=np.dot(F,np.array(sol['x'][:N]).flatten())
    f_opt_max=np.max(np.abs(f_opt))
    ax3.plot(k,20*np.log10(np.abs(f_opt))-20*np.log10(f_opt_max))
    ax3.set_title('DFT sample points as seen by optimization algorithm')
    plt.show()
else:
    print 'no solution found'
    print 'status: %s' % (sol['status'],)
