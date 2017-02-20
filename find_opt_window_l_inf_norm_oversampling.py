# This doesn't work...

import numpy as np
import cvxopt
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import ddm
import sys

# Window length (non-zero part)
N=512

# Oversampling
os=16

# L is total length of window including zero-padding
L=os*N

# "Half" Fourier basis (cosine part only)
n=np.arange(N)
l=np.arange(L)
k=np.arange(L)
F_=np.cos(-2*np.pi*np.outer(k,l)/L)
#F=F_[:,((L-N)/2):(-(L-N)/2)]
F=np.concatenate((F_[:,:(N/2)],F_[:,-(N/2):]),axis=1)

## Show it is correct
#w_test,dw=ddm.w_dw_sum_cos(N,a='c1-nuttall-4')
##W_test=np.dot(F,w_test)
#w_test_padded=np.concatenate((w_test,np.zeros(L-N)))
#w_test_padded=np.concatenate((w_test_padded[(N/2):],w_test_padded[:(N/2)]))
##W_test=np.inner(F_,np.concatenate((np.zeros((L-N)/2),w_test,np.zeros((L-N)/2))))
##W_test=np.dot(F_,w_test_padded)
#W_test=np.dot(F,np.concatenate((w_test_padded[:(N/2)],w_test_padded[-(N/2):])))
#plt.plot(k/float(os),20*np.log10(W_test))
#plt.show()

d=np.zeros((F.shape[0],1))
d[0]=1.

# Number of cosines to sum
M=4
m=np.arange(M)
# Cosines to sum to get window
C=np.cos(2*np.pi*np.outer(n,m)/N)*np.power(1.,np.arange(M))


# The bin from which to start checking the sidelobes
R=0

# We will solve a LP to find a solution
# See http://cvxopt.org/userguide/coneprog.html#linear-programming for the form

# The c vector of the LP
c=np.concatenate((np.zeros(N+M),[1]))

# G inequality matrix
#G=np.zeros((2*(L/2+1-R)+1,N+M+1))
#G[:(L/2+1-R),:]=np.concatenate((F[R:(L/2+1),:],np.zeros(((L/2+1)-R,M)),np.ones((L/2+1-R,1))),axis=1)
#G[(L/2+1-R):-1,:]=np.concatenate((-1*F[R:(L/2+1),:],np.zeros(((L/2+1)-R,M)),np.ones((L/2+1-R,1))),axis=1)
#G[-1,-1]=-1.
G=np.zeros((2*(N-2*R)+1,N+M+1))
G[:(N-2*R),:]=np.concatenate((F[R+1:-R,:],np.zeros((N-2*R,M)),np.ones((N-2*R,1))),axis=1)
G[(N-2*R):-1,:]=np.concatenate((-1*F[R+1:-R,:],np.zeros((N-2*R,M)),np.ones((N-2*R,1))),axis=1)
G[-1,-1]=-1.
G=np.concatenate((
        np.concatenate((np.identity(N),np.zeros((N,M)),np.zeros((N,1))),axis=1),
        np.concatenate((-1*np.identity(N),np.zeros((N,M)),np.zeros((N,1))),axis=1),
        G
    ),
    axis=0)

# h inequality vector
#h=np.concatenate((d[R:(L/2+1)],-1*d[R:(L/2+1)],np.zeros((1,1))),axis=0)
h=np.concatenate((d[R+1:-R],-1*d[R+1:-R],np.zeros((1,1))),axis=0)
h=np.concatenate((np.ones((N,1)),np.ones((N,1)),h),axis=0)

# The A equality matrix
A=np.concatenate((
        np.concatenate((-1*np.identity(N),C,np.zeros((N,1))),axis=1),
        np.concatenate((np.zeros((1,N)),np.ones((1,M)),np.zeros((1,1))),axis=1),
        np.concatenate((np.zeros((1,N)),C[0,:].reshape((1,M)),np.zeros((1,1))),axis=1),
        np.concatenate((F[0,:].reshape((1,N)),np.ones((1,M)),np.zeros((1,1))),axis=1)),
    axis=0)

# The b equality vector
b=np.zeros((N+3,1))
b[N]=1
b[N+2]=1

c=cvxopt.matrix(c)
A=cvxopt.matrix(A)
b=cvxopt.matrix(b)
G=cvxopt.matrix(G)
h=cvxopt.matrix(h)

sol=cvxopt.solvers.lp(c,G=G,h=h,A=A,b=b)

if sol['status'] == 'optimal':
    print 'solution found'
    print 'x='
    print sol['x'][:N]
    print 'F*x='
    print np.dot(F,np.array(sol['x'][:N]))
    print 'C*a='
    print np.dot(C,np.array(sol['x'][N:N+M]))
    print 'a='
    print sol['x'][N:N+M]
    print 't='
    print sol['x'][-1]
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
