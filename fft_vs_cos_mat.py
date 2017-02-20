# We are interested in Fourier transforming even functions for which the FT is
# purely real. Therefore should suffice to only use cosine basis functions,
# right?

import ddm
import numpy as np
import matplotlib.pyplot as plt

M = 8
N = 2*M
w,dw=ddm.w_dw_sum_cos(M,a='hanning')
w_padded=np.concatenate((w,np.zeros(N-M)))
w_padded=np.concatenate((w_padded[(M/2):],w_padded[:(M/2)]))
#W=np.fft.fft(np.concatenate((np.zeros((N-M)/2),w,np.zeros((N-M)/2))))
W=np.fft.fft(w_padded)
k=np.arange(N)
n=np.arange(N)
F=np.cos(-2*np.pi*np.outer(k,n)/N)
#W_=np.dot(F,np.concatenate((np.zeros((N-M)/2),w,np.zeros((N-M)/2))))
W_=np.dot(F,w_padded)
fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2)
ax1.plot(k,np.real(W))
ax1.set_title('DFT')
ax2.plot(k,W_)
ax2.set_title('Dab cosine')
ax3.plot(np.concatenate((k[0:1],k[1::2])),np.concatenate((np.abs(W[0:1]),np.abs(W[1::2]))))
ax3.set_title('DFT, origin and midpoints')
ax4.set_title('dab cosine, origin and midpoints')
ax4.plot(np.concatenate((k[0:1],k[1::2])),np.concatenate((np.abs(W_[0:1]),np.abs(W_[1::2]))))
plt.show()
