import ddm
import numpy as np
import matplotlib.pyplot as plt

# Sample rate (Hz)
F_s=16000
# Window length (ms)
W_l_ms=32
# Window length (samples)
W_l=F_s*(W_l_ms/1000.)
# Time indices
t=np.arange(W_l)/float(F_s)

# Parameter ranges
# Amplitude part
a0_r=[0,1.]
a1_r=[-100.,100.]
a2_r=[-1.e4,1.e3]
a3_r=[-1.e5,1.e5]
# Phase part
ph0_r=[-np.pi,np.pi]
ph1_r=[0.,np.pi*F_s]
#ph1_r=[0.,0.]
ph2_r=[-1.e4,1.e4]
#ph2_r=[1.e4,1.e4]
ph3_r=[-1.e6,1.e6]

a0=np.random.uniform(a0_r[0],a0_r[1])
a1=np.random.uniform(a1_r[0],a1_r[1])
a2=np.random.uniform(a2_r[0],a2_r[1])
ph0=np.random.uniform(ph0_r[0],ph0_r[1])
ph1=np.random.uniform(ph1_r[0],ph1_r[1])
ph2=np.random.uniform(ph2_r[0],ph2_r[1])

# Synthesize signal
# Real argument
arg_re=np.polyval([a2,a1,a0],t)
# Imaginary argument
arg_im=np.polyval([ph2,ph1,ph0],t)
# Analytic signal
x=np.exp(arg_re+1j*arg_im)
# Power of analytic signal
p_x=np.sum(x**2.)/float(W_l)
## Compute DFT without noise to find maximum
#X=np.fft.fft(x*w)
(n,X)=ddm.plot_sig_dft_c(x,show=False)
kma0=np.argmax(np.abs(X))
plt.plot(n[kma0],20*np.log10(np.abs(X[kma0])),'o')
plt.show()
