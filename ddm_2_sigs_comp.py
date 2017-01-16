# Compare the accuracy of the DDM using different atoms (windows)
# when using a mixture of 2 signals.
# Results are plotted as parameter error versus distance in number of bins
#
# TODO: Results don't seem that great, but they really should be no?


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
N_eval=10000


# SNRs to test
snrs=np.arange(-2,11)*(10.)

# Errors
errs=dict()
keys=['hann','c1-nuttall-4','c1-nuttall-3','c1-blackman-4']
key_clrs=['b','g','r','m']
for k,c in zip(keys,key_clrs):
    errs[k]=dict()
    # Error aggregates
    errs[k]['agg']=dict()
    # Error counts
    errs[k]['cnt']=dict()
    for k_ in ['a0_err', 'a1_err', 'a2_err', 'ph0_err', 'ph1_err', 'ph2_err']:
        errs[k]['agg'][k_]=dict()
        errs[k]['cnt'][k_]=dict()
    errs[k]['clr']=c

for k in errs.keys():

    # Analysis window, and derivative
    w,dw=ddm.w_dw_sum_cos(W_l,k,norm=False)
    
    snr = 80
    n = 0

    nlsqerrs=0
    
    # Order 2 model
    while (n < N_eval):
        x_tot=np.zeros(int(W_l),dtype='complex')
        kma0=[-1,-1]
        alphas=[None,None]
        true_alphas=[dict(),dict()]
        for n_sig in xrange(2):
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
            # Compute DFT without noise to find maximum
            X=np.fft.fft(x*w)
            kma0[n_sig]=np.argmax(np.abs(X))
            x_tot+=x
            true_alphas[n_sig]['a0']=a0
            true_alphas[n_sig]['a1']=a1
            true_alphas[n_sig]['a2']=a2
            true_alphas[n_sig]['ph0']=ph0
            true_alphas[n_sig]['ph1']=ph1
            true_alphas[n_sig]['ph2']=ph2
        # Power of analytic signal
        p_x_tot=np.sum(x_tot**2.)/float(W_l)
        # Power of noise
        p_no=p_x_tot*(10.**(-1*snr/10.))
        # Gain to synthesize noise of desired power
        g_no=np.sqrt(p_no)
        # Add noise
        y=x_tot+g_no*np.random.standard_normal(int(W_l))
        # Estimate parameters
        alph=ddm.ddm_p2_1_3(y,w,dw,kma0[0])
        if (alph == None):
            # If there was a problem solving the system of equations, just
            # ignore and try again
            continue
            nlsqerrs+=1
        alphas[0]=alph
        alph=ddm.ddm_p2_1_3(y,w,dw,kma0[1])
        if (alph == None):
            # If there was a problem solving the system of equations, just
            # ignore and try again
            continue
            nlsqerrs+=1
        alphas[1]=alph
        # Record errors
        # DDM estimation is done with normalized frequency, so correct
        # coefficients
        err_=dict()
        err_['a0_err']=0
        err_['a1_err']=0
        err_['a2_err']=0
        err_['ph0_err']=0
        err_['ph1_err']=0
        err_['ph2_err']=0
        for alpha, ta in zip(alphas,true_alphas):
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
            err_['a0_err']+=np.abs(a0-a0_)
            err_['a1_err']+=np.abs(a1-a1_)
            err_['a2_err']+=np.abs(a2-a2_)
            # Find minimum distance on circle
            err_['ph0_err']+=min([(ph0-ph0_)%(2.*np.pi),(ph0_-ph0)%(2.*np.pi)])
            # We do not allow negative frequencies (TODO: is this okay?)
            err_['ph1_err']+=np.abs(ph1-np.abs(ph1_))
            err_['ph2_err']+=np.abs(ph2-ph2_)
        # Error is average estimation error
        err_['a0_err']/=2.
        err_['a1_err']/=2.
        err_['a2_err']/=2.
        err_['ph0_err']/=2.
        err_['ph1_err']/=2.
        err_['ph2_err']/=2.
        bindiff=max(kma0)-min(kma0)
        for ek in err_.keys():
            try:
                errs[k]['agg'][ek][bindiff]+=err_[ek]
            except KeyError:
                errs[k]['agg'][ek][bindiff]=err_[ek]
            try:
                errs[k]['cnt'][ek][bindiff]+=1
            except KeyError:
                errs[k]['cnt'][ek][bindiff]=1

        n += 1

    print 'nerrs: %d for snr %f' % (nlsqerrs,snr)

fig, axarr = plt.subplots(6,1)

for k in errs.keys():
    for ek,pl in zip(errs[k]['agg'].keys(),axarr[:]):
        e_v=np.zeros(len(errs[k]['agg'][ek].keys()),dtype='double')
        e_bd=np.zeros(len(errs[k]['agg'][ek].keys()),dtype='double')
        for i,bd in enumerate(errs[k]['agg'][ek].keys()):
            e_v[i] = errs[k]['agg'][ek][bd] / float(errs[k]['cnt'][ek][bd])
            e_bd[i]=bd
        e_bd_srt_i=np.argsort(e_bd)

        pl.plot(e_bd[e_bd_srt_i],np.log10(e_v[e_bd_srt_i]),c=errs[k]['clr'],label=k)
        pl.set_title(ek)

axarr[0].legend(fontsize=10)

plt.show()
