# Compare the accuracy of the DDM using different atoms (windows) and in various
# SNRs.

# TODO: Normalize by Cramer-Rao bound?

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
N_eval=100


# SNRs to test
snrs=np.arange(-2,11)*(10.)

# Errors
errs=dict()
keys=['hann','c1-nuttall-4','c1-nuttall-3','c1-blackman-4']
key_clrs=['b','g','r','m']
for k,c in zip(keys,key_clrs):
    errs[k]=dict()
    # Amplitude part
    errs[k]['a']=[[],[],[]]
    # Phase part
    errs[k]['ph']=[[],[],[]]
    errs[k]['clr']=c

for k in errs.keys():

    # Analysis window, and derivative
    w,dw=ddm.w_dw_sum_cos(W_l,k,norm=False)
    
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

        nlsqerrs=0
        
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
            p_x=np.sum(x*np.conj(x))/float(W_l)
            # Compute DFT without noise to find maximum
            X=np.fft.fft(x*w)
            kma0=np.argmax(np.abs(X))
            # Power of noise
            p_no=p_x*(10.**(-1*snr/10.))
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
                nlsqerrs+=1
            # Record errors
            # DDM estimation is done with normalized frequency, so correct
            # coefficients
            a0_=np.real(alpha[0][0])
            a1_=np.real(alpha[0][1])*F_s
            a2_=np.real(alpha[0][2])*(F_s**2.)
            ph0_=np.imag(alpha[0][0])
            while (ph0_ > ph0_r[1]):
                ph0_ -= 2.*np.pi
            while (ph0 < ph0_r[0]):
                ph0_ += 2.*np.pi
            ph1_=np.imag(alpha[0][1])*F_s
            ph2_=np.imag(alpha[0][2])*(F_s**2.)
            a0_err+=np.abs(a0-a0_)
            a1_err+=np.abs(a1-a1_)
            a2_err+=np.abs(a2-a2_)
            # Find minimum distance on circle
            ph0_err+=min([(ph0-ph0_)%(2.*np.pi),(ph0_-ph0)%(2.*np.pi)])
            # We do not allow negative frequencies (TODO: is this okay?)
            ph1_err+=np.abs(ph1-np.abs(ph1_))
            ph2_err+=np.abs(ph2-ph2_)
            n += 1

        print 'nerrs: %d for snr %f' % (nlsqerrs,snr)
    
        errs[k]['a'][0].append(a0_err/float(N_eval))
        errs[k]['a'][1].append(a1_err/float(N_eval))
        errs[k]['a'][2].append(a2_err/float(N_eval))
        errs[k]['ph'][0].append(ph0_err/float(N_eval))
        errs[k]['ph'][1].append(ph1_err/float(N_eval))
        errs[k]['ph'][2].append(ph2_err/float(N_eval))

fig, axarr = plt.subplots(3,2)

for k in errs.keys():
    for i,pl in zip(xrange(len(errs[k]['a'])),axarr[:,0]):
        pl.plot(snrs,np.log10(errs[k]['a'][i]),c=errs[k]['clr'],label=k)
        pl.set_title('a_%d' % (i,))
    for i,pl in zip(xrange(len(errs[k]['ph'])),axarr[:,1]):
        pl.plot(snrs,np.log10(errs[k]['ph'][i]),c=errs[k]['clr'],label=k)
        pl.set_title('ph_%d' % (i,))

axarr[0,0].legend(fontsize=10)

plt.show()
