# Compare the accuracy of the DDM using different atoms (windows) and in various
# SNRs.

# TODO: Normalize by Cramer-Rao bound?

import ddm
import numpy as np
import matplotlib.pyplot as plt

plt.rc('text',usetex=True)
plt.rc('font',**ddm.FONT_OPT['dafx'])

show_plots=False

# Sample rate (Hz)
F_s=16000
# Window length (ms)
W_l_ms=32
# Window length (samples)
W_l=F_s*(W_l_ms/1000.)
# Time indices
t=np.arange(W_l)/float(F_s)
# Number of bins to use in estimation
R=3

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

# SNRs to test
snrs=np.arange(-2,17)*(10.)

# Errors
errs=dict()
keys=['crlb','hann','c1-nuttall-4','c1-nuttall-3','prolate-0.008-approx-5']
key_clrs=['LightGrey']+['k' for _ in xrange(4)]
key_ls=['solid','solid','dotted','dashed','dashdot']
key_labels=['C','H','N4','N3','P5']
for k,c,ls,lab in zip(keys,key_clrs,key_ls,key_labels):
    errs[k]=dict()
    # Amplitude part
    errs[k]['a']=[[],[],[]]
    # Phase part
    errs[k]['ph']=[[],[],[]]
    # Plotting descriptors
    errs[k]['plot_dict']={'c':c,'ls':ls,'label':lab}

for k in errs.keys():

    # Analysis window, and derivative
    if k != 'crlb':
        w,dw=ddm.w_dw_sum_cos(W_l,k,norm=False)
    
    for snr in snrs:
        
        # Amplitude part
        # mean aggregate
        a0_err_mean_ag=0
        a1_err_mean_ag=0
        a2_err_mean_ag=0
        # 2nd moment aggregate (to compute variance at the end)
        a0_err_2mom_ag=0
        a1_err_2mom_ag=0
        a2_err_2mom_ag=0
        # Phase part
        # mean aggregate
        ph0_err_mean_ag=0
        ph1_err_mean_ag=0
        ph2_err_mean_ag=0
        # 2nd moment aggregate (to compute variance at the end)
        ph0_err_2mom_ag=0
        ph1_err_2mom_ag=0
        ph2_err_2mom_ag=0
    
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
            # Power of noise
            p_no=p_x*(10.**(-1*snr/10.))
            if k == 'crlb':
                # Don't do analysis, just compute CRLB
                v_crlb=ddm.crlb_pq(np.array([a0,a1,a2]),t,p_no)
                # In this case it is an mean variance aggregate (not the mean
                # aggregate)
                a0_err_mean_ag+=v_crlb[0]
                a1_err_mean_ag+=v_crlb[1]
                a2_err_mean_ag+=v_crlb[2]
                # The Fischer matrices are equivalent for the log-amplitude and
                # phase parameters and so their CRLBs are the same
                ph0_err_mean_ag+=v_crlb[0]
                ph1_err_mean_ag+=v_crlb[1]
                ph2_err_mean_ag+=v_crlb[2]
            else:
                # Gain to synthesize noise of desired power
                g_no=np.sqrt(p_no)
                # Compute DFT without noise to find maximum
                X=np.fft.fft(x*w)
                kma0=np.argmax(np.abs(X))
                # Add noise
                y=x+g_no*np.random.standard_normal(int(W_l))
                # Estimate parameters
                alpha=ddm.ddm_p2_1_R(y,w,dw,kma0,R)
    #            alpha=ddm.ddm_p2_1_3(y,w,dw,kma0)
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
                while (ph1_ > ph1_r[1]):
                    ph1_ -= np.pi*F_s
                while (ph1_ < ph1_r[0]):
                    ph1_ += np.pi*F_s
                ph2_=np.imag(alpha[0][2])*(F_s**2.)
                a0_err_mean_ag+=np.abs(a0-a0_)
                a1_err_mean_ag+=np.abs(a1-a1_)
                a2_err_mean_ag+=np.abs(a2-a2_)
                a0_err_2mom_ag+=np.power(a0-a0_,2.)
                a1_err_2mom_ag+=np.power(a1-a1_,2.)
                a2_err_2mom_ag+=np.power(a2-a2_,2.)
                # Find minimum distance on circle
                ph0_err=min([(ph0-ph0_)%(2.*np.pi),(ph0_-ph0)%(2.*np.pi)])
                # Frequency error found on circle too (because of sampling theorem)
                ph1_err=min([(ph1-ph1_)%(np.pi*F_s),(ph1_-ph1)%(np.pi*F_s)])
                ph2_err=np.abs(ph2-ph2_)
                ph0_err_mean_ag+=ph0_err
                ph1_err_mean_ag+=ph1_err
                ph2_err_mean_ag+=ph2_err
                ph0_err_2mom_ag+=np.power(ph0_err,2.)
                ph1_err_2mom_ag+=np.power(ph1_err,2.)
                ph2_err_2mom_ag+=np.power(ph2_err,2.)
            n += 1

        print 'nerrs: %d for snr %f' % (nlsqerrs,snr)
    
        if (k == 'crlb'):
            errs[k]['a'][0].append(a0_err_mean_ag/float(N_eval))
            errs[k]['a'][1].append(a1_err_mean_ag/float(N_eval))
            errs[k]['a'][2].append(a2_err_mean_ag/float(N_eval))
            errs[k]['ph'][0].append(ph0_err_mean_ag/float(N_eval))
            errs[k]['ph'][1].append(ph1_err_mean_ag/float(N_eval))
            errs[k]['ph'][2].append(ph2_err_mean_ag/float(N_eval))
        else:
            errs[k]['a'][0].append((a0_err_2mom_ag/float(N_eval)) +
                    np.power(a0_err_mean_ag/float(N_eval),2.))
            errs[k]['a'][1].append((a1_err_2mom_ag/float(N_eval)) +
                    np.power(a1_err_mean_ag/float(N_eval),2.))
            errs[k]['a'][2].append((a2_err_2mom_ag/float(N_eval)) +
                    np.power(a2_err_mean_ag/float(N_eval),2.))
            errs[k]['ph'][0].append((ph0_err_2mom_ag/float(N_eval)) +
                    np.power(ph0_err_mean_ag/float(N_eval),2.))
            errs[k]['ph'][1].append((ph1_err_2mom_ag/float(N_eval)) +
                    np.power(ph1_err_mean_ag/float(N_eval),2.))
            errs[k]['ph'][2].append((ph2_err_2mom_ag/float(N_eval)) +
                    np.power(ph2_err_mean_ag/float(N_eval),2.))

fig, axarr = plt.subplots(3,2,sharex=True)

a_mins=[0,0,0]
ph_mins=[0,0,0]
a_maxs=[0,0,0]
ph_maxs=[0,0,0]

for k in errs.keys():
    for i,pl in zip(xrange(len(errs[k]['a'])),axarr[:,0]):
        pl.plot(snrs,np.log10(errs[k]['a'][i]),**errs[k]['plot_dict'])
        pl.set_title('$\Re\{a_%d\}$' % (i,))
        pl.set_ylabel('Error variance ($\log_{10}$)')
        if (i == 2):
            pl.set_xlabel('SNR (dB)')
        pl.set_xlim([snrs[0],snrs[-1]])
        if (np.max(np.log10(errs[k]['a'][i])) > a_maxs[i]):
            a_maxs[i]=np.max(np.log10(errs[k]['a'][i]))
        if (np.min(np.log10(errs[k]['a'][i])) < a_mins[i]):
            a_mins[i]=np.min(np.log10(errs[k]['a'][i]))

    for i,pl in zip(xrange(len(errs[k]['ph'])),axarr[:,1]):
        pl.plot(snrs,np.log10(errs[k]['ph'][i]),**errs[k]['plot_dict'])
        pl.set_title('$\Im\{a_%d\}$' % (i,))
        if (i == 2):
            pl.set_xlabel('SNR (dB)')
        pl.set_xlim([snrs[0],snrs[-1]])
        if (np.max(np.log10(errs[k]['ph'][i])) > ph_maxs[i]):
            ph_maxs[i]=np.max(np.log10(errs[k]['ph'][i]))
        if (np.min(np.log10(errs[k]['ph'][i])) < ph_mins[i]):
            ph_mins[i]=np.min(np.log10(errs[k]['ph'][i]))

for i,pl in zip(xrange(len(errs[k]['a'])),axarr[:,0]):
    pl.set_ylim([a_mins[i],a_maxs[i]])
for i,pl in zip(xrange(len(errs[k]['ph'])),axarr[:,1]):
    pl.set_ylim([ph_mins[i],ph_maxs[i]])

fig.suptitle(
    'Parameter estimation error variance in various SNR',
    # hacky way to get title size
    fontsize=axarr[0,0].title.get_fontproperties().get_size_in_points())

for pl in axarr.flatten():
    pl.title.get_fontproperties().set_size(10)

axarr[0,0].legend(fontsize=10,loc='lower left')

fig.set_size_inches(10,7)
plt.savefig('paper/ddm_snr_win_comp.eps')

with open('paper/ddm_snr_win_comp_defs.txt','w') as f:
    f.write('\\newcommand{\Ksnr}{%d}\n' % (N_eval,))

if (show_plots):
    plt.show()
