import ddm
import numpy as np

N=8
n=np.arange(N)
k=np.arange(N)

w=ddm.w_dw_sum_cos(8,'hann')[0]
f_fft=np.real(np.fft.fft(w))

# vectors of real part only
C=np.cos(2*np.pi*np.outer(k,n)/N)

f_ct=np.inner(C,w)
#f_ct/=f_ct[0]

print f_fft
print f_ct
