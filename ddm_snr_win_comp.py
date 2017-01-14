# Compare the accuracy of the DDM using different atoms (windows) and in various
# SNRs.

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
ph2_r=[-1.e4,1.e4]
ph3_r=[-1.e6,1.e6]

# Number of evaluations to perform
N_eval=1000

# Analysis window, and derivative
w,dw=ddm.w_dw_sum_cos(W_l,'hann')

# SNRs to test
snrs=np.arange(-2,11)*(-10.)

# Amplitude part
a0_errs=[]
a1_errs=[]
a2_errs=[]
a3_errs=[]
# Phase part
ph0_errs=[]
ph1_errs=[]
ph2_errs=[]
ph3_errs=[]

for snr in snrs:
    
    # Amplitude part
    a0_err=0.
    a1_err=0.
    a2_err=0.
    a3_err=0.
    # Phase part
    ph0_err=0.
    ph1_err=0.
    ph2_err=0.
    ph3_err=0.

    n = 0
    
    # Order 2 model
    while (n < N_eval):
        # Draw random parameters
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
        # Compute DFT without noise to find maximum
        X=np.fft.fft(x*w)
        kma0=np.argmax(np.abs(X))
        # Power of noise
        p_no=p_x*(10.**(snr/10.))
        # Gain to synthesize noise of desired power
        g_no=np.sqrt(p_no)
        # Add noise
        y=x+g_no*np.random.standard_normal(int(W_l))
        # Estimate parameters
        alpha=ddm.ddm_p2_1_3(y,w,dw,kma0)
        if (alpha == None):
            # If there was a problem solving the system of equations, just
            # ignore and try again
            continue
        # Record errors
        a0_=np.real(alpha[0][0])
        a1_=np.real(alpha[0][1])
        a2_=np.real(alpha[0][2])
        ph0_=np.imag(alpha[0][0])
        ph1_=np.imag(alpha[0][1])
        ph2_=np.imag(alpha[0][2])
        a0_err+=np.abs(a0-a0_)
        a1_err+=np.abs(a1-a1_)
        a2_err+=np.abs(a2-a2_)
        ph0_err+=np.abs(ph0-ph0_)
        ph1_err+=np.abs(ph1-ph1_)
        ph2_err+=np.abs(ph2-ph2_)
        n += 1

    a0_errs.append(a0_err/float(N_eval))
    a1_errs.append(a1_err/float(N_eval))
    a2_errs.append(a2_err/float(N_eval))
    ph0_errs.append(ph0_err/float(N_eval))
    ph1_errs.append(ph1_err/float(N_eval))
    ph2_errs.append(ph2_err/float(N_eval))

plt.plot(snrs,ph1_errs)
plt.show()
