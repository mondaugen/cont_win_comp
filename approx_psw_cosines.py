import ddm
import sigmod as sm
import numpy as np
import matplotlib.pyplot as plt
import cvxopt

show_plot=False

# Window length
N=512
n=np.arange(N)

# Oversampling (for plotting)
os=4

# Number of cosines to sum
M=47
m=np.arange(M)
# Cosines to sum to get window
C=np.cos(2*np.pi*np.outer(n,m)/N)*np.power(-1.,np.arange(M))

# Prolate spheroidal window
W=0.77
v=ddm.psw_design(N+1,0.77)
# Make window positive
if (np.abs(np.min(v)) > np.abs(np.max(v))):
    v *= -1.
# Make window maximum equal to 1
v/=np.max(v)

# Solve for as that approximate v
# Constraint matrix
A=np.concatenate((np.ones((1,M)),C[0,:].reshape((1,M))),axis=0)
b=np.array([[1.],[0.]])

# Prepare matrices to solve constrained least-squares
P=2.*np.dot(C.T,C)
q=-2.*np.dot(C.T,v[:-1].reshape(N,1))

A=cvxopt.matrix(A)
b=cvxopt.matrix(b)
P=cvxopt.matrix(P)
q=cvxopt.matrix(q)

a=cvxopt.solvers.qp(P,q,A=A,b=b)['x']
print 'a coefficients'
print a
a=np.array(a).flatten()

# save to file
np.savetxt('/tmp/prolate_approx_W=0.77_M=%d.txt' % (M,),a)

# The sum-of-cosine window
w=ddm.w_dw_sum_cos(N,a)[0]

# Magnitude spectra
N_F=N*os
v_F=np.fft.fft(v,N_F)
w_F=np.fft.fft(np.concatenate((w.reshape((1,N)),w[0].reshape((1,1))),axis=1),N_F).flatten()
v_F=20*np.log10(np.abs(v_F))
w_F=20*np.log10(np.abs(w_F))
v_F-=np.max(v_F)
w_F-=np.max(w_F)
v_F=np.concatenate((v_F[N_F:],v_F[:N_F]))
w_F=np.concatenate((w_F[N_F:],w_F[:N_F]))
bins_F=np.arange(0,N,1./os)

fig1,axs=plt.subplots(2,2)
axs[0,0].plot(bins_F,v_F)
axs[0,0].set_title('prolate spheroidal window, magnitude spectrum')
axs[1,0].plot(bins_F,w_F)
axs[1,0].set_title('sum-of-cosine approx. window, magnitude spectrum')

axs[0,1].plot(n,v[:-1])
axs[0,1].set_title('prolate spheroidal window, time-domain')
axs[1,1].plot(n,w)
axs[1,1].set_title('sum-of-cosine approx. window, time-domain')

if (show_plot):
    plt.show()

