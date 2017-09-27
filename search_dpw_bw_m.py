import numpy as np
import numpy.linalg
import ddm
import matplotlib.pyplot as plt
import sigmod as sm

plt.rc('text',usetex=True)
plt.rc('font',**ddm.FONT_OPT['dafx'])

# Search for number of cosines that best approximate window

# Length of window
N=513
# Length of DFT (controls zero-padding)
N_F=1024
#n_F=np.arange(N_F)
#N=513
# Design coefficients
# This range found empirically to give peak side-lobes close to desired one
W=np.arange(0.005,0.011,0.001)
#W=np.array([0.009])
#W=np.array([0.009])
# Ratio of max to peak sidelobe (dB)
a=np.zeros(len(W))
# The 3db-bandwidth desired
#bw_des=2.
bw_des=1.91
#W=.8
i=np.arange(N,dtype='double')
# Number of cosines to use in approximation
M_w=np.arange(2,50)
# Constraints to use when approximating (just continuous)
c_w={'c1'}

w_close=None
bw_diff_close=100.
v_close=None
v_F_close=None
for n,W_ in enumerate(W):
    v=ddm.psw_design(N,W_)
    # Remove endpoint because it should be the same as the first point
    v=v[:-1]
    # Make window positive
    if (np.abs(np.min(v)) > np.abs(np.max(v))):
        v *= -1.
    # Make window maximum equal to 1
    v/=np.max(v)
    try:
        k_opt=ddm.bw_bins(v,k_hint=bw_des/2.)
    except RuntimeError:
        a[n]=None
        continue
    bw_diff=np.abs(2.*k_opt-bw_des)
    #(ma,mai)=sm.lextrem(v_F)
    #ma_srt=np.sort(ma.flatten())
    #a[n]=ma_srt[-1]-ma_srt[-2]
    #a_diff=np.abs(a_des-a[n])
    a[n]=np.abs(k_opt*2.)
    if (bw_diff < bw_diff_close):
        bw_diff_close=bw_diff
        w_close=W_
        v_close=v
        v_F=20.*np.log10(np.abs(np.fft.fft(v,N_F)))
        v_F_close=v_F

# We define half the mainlobe width to be the distance from the peak to the
# first inflection point in the power spectrum after the peak
mlb_radius=sm.lextrem(v_F_close,comp='min')[1][0][0]/(N_F/(N-1))
#v_F_close=np.concatenate((v_F_close[N_F/2:],v_F_close[:N_F/2]))
a=[None if a_ > 100 else a_ for a_ in a]
i_a_valid=[i for (i,val) in enumerate(a) if val != None]
a=[a[i] for i in i_a_valid]
W=[W[i] for i in i_a_valid]

print "bw_diff_close: %f" % (bw_diff_close,)
print "v_close end point: %g" % (v_close[0],)
print "w_close: %g" % (w_close,)

# Approximate designed window using sum of cosines
# Find M_w that best approximates main lobe
w_aprx_err=float('inf')
M_w_opt=None
w_aprx_F=None
w_aprx=None
for M_w_ in M_w:
    a_w_=ddm.cos_approx_win(v,M_w_,c_w)
    w_aprx_=ddm.w_dw_sum_cos(N-1,a=a_w_)[0]
    w_aprx_F_=20.*np.log10(np.abs(np.fft.fft(w_aprx_,N_F)))
    w_err_idxs=np.arange(0,mlb_radius)
    _w_err=np.sum(np.abs(w_aprx_F_[w_err_idxs]-v_F_close[w_err_idxs]))
    if (_w_err < w_aprx_err):
        w_aprx_err=_w_err
        M_w_opt=M_w_
        w_aprx_F=w_aprx_F_
        w_aprx=w_aprx_
        a_w=a_w_

print 'M_w_opt: %d' % (M_w_opt,)

# Plot windows
#fig, (ax1,ax2) = plt.subplots(2,1)
#ax1.plot(W,a)
#ax1.set_title("W versus 3db bandwidth")
#
# Plot nuttall window for comparison
w_nuttall=ddm.w_dw_sum_cos(N-1,a='c1-nuttall-4')[0]
w_F_nuttall=20.*np.log10(np.abs(np.fft.fft(w_nuttall,N_F)))
##w_F_nuttall=np.concatenate((w_F_nuttall[N_F/2:],w_F_nuttall[:N_F/2]))

# Remove local minima from plot for clarity
plt_thresh_db=-250
w_aprx_F_plt=w_aprx_F-np.max(w_aprx_F)
waFp_i=[i for i,w_ in enumerate(w_aprx_F_plt) if w_ > plt_thresh_db]
w_aprx_F_plt=w_aprx_F_plt[waFp_i]

w_F_nuttall_plt=w_F_nuttall-np.max(w_F_nuttall)
wFnp_i=[i for i,w_ in enumerate(w_F_nuttall_plt) if w_ > plt_thresh_db]
w_F_nuttall_plt=w_F_nuttall_plt[wFnp_i]

#leg_nuttall,=ax2.plot(np.arange(N_F)[wFnp_i]/(float(N_F)/(N-1)),
#        w_F_nuttall_plt,c='r',label="Nuttall")
#leg_prolate,=ax2.plot(np.arange(N_F)/(float(N_F)/(N-1)),
#        v_F_close-np.max(v_F_close),c='b',label="Prolate")
#leg_aprx,=ax2.plot(np.arange(N_F)[waFp_i]/(float(N_F)/(N-1)),
#        w_aprx_F_plt,c='g',label="Prolate Approximation")
#ax2.set_title("nuttall versus prolate and its approximation")
#ax2.legend(handles=[leg_nuttall,leg_prolate,leg_aprx])

n_figs=2
figs,axs=plt.subplots(1,2)
plot_funs=['axs[fignum].plot','axs[fignum].plot']
plot_args={
    "Nuttall" : [{'ls':'dotted'} for _ in xrange(n_figs)],
    "Prolate" : [{'ls':'solid'} for _ in xrange(n_figs)],
    "Prolate Approximation" : [{'ls':'dashed'} for _ in xrange(n_figs)]
}
for pa in plot_args.keys():
    for pa_ in plot_args[pa]:
        pa_['c']='k'

fig_titles=['Nuttall, Prolate and its approximation:\n asymptotic behaviour',
    'Nuttall, Prolate and its approximation:\n main lobe']
fig_short_titles=['asymptotic','mainlobe']
fig_xlims=[[(N-1.)/N_F,(N-1)/2],[0,(N-1)/16]]
fig_ylims=[[-250,0],[-160,0]]
fig_xlabels=['Bin number' for _ in xrange(n_figs)]
fig_ylabels=['Power (dB)','']
fig_xscales=['log','linear']
fig_xscales_kwargs=[{'basex':2},{}]
leg_nuttall =None
leg_prolate =None
leg_aprx    =None
#for fig in figs:
for fignum in xrange(n_figs):
    leg_nuttall,=eval(plot_funs[fignum])(np.arange(N_F)[wFnp_i]/(float(N_F)/(N-1)),
            w_F_nuttall_plt,label="Nuttall",**plot_args["Nuttall"][fignum])
    leg_prolate,=eval(plot_funs[fignum])(np.arange(N_F)/(float(N_F)/(N-1)),
            v_F_close-np.max(v_F_close),label="Prolate",**plot_args["Prolate"][fignum])
    leg_aprx,=eval(plot_funs[fignum])(np.arange(N_F)[waFp_i]/(float(N_F)/(N-1)),
            w_aprx_F_plt,label="Prolate Approximation",
            **plot_args["Prolate Approximation"][fignum])
    axs[fignum].set_title(fig_titles[fignum])
    axs[fignum].set_xlim(fig_xlims[fignum])
    axs[fignum].set_ylim(fig_ylims[fignum])
    axs[fignum].set_xlabel(fig_xlabels[fignum])
    axs[fignum].set_ylabel(fig_ylabels[fignum])
    axs[fignum].set_xscale(fig_xscales[fignum],**fig_xscales_kwargs[fignum])
    axs[fignum].grid(ls='dotted')

axs[1].legend(fontsize=10,handles=[leg_nuttall,leg_prolate,leg_aprx])
figs.set_size_inches(10,4)
figs.savefig('paper/search_dpw_bw_m.eps')

print "w_nuttall end point: %g" % (w_nuttall[0],)

k_opt_w_aprx=ddm.bw_bins(w_aprx,k_hint=bw_des/2.)*2.
print "w_aprx 3db bandwidth: %g" % (k_opt_w_aprx,)

# save coefficients to file
np.savetxt('/tmp/prolate_approx_W=%1.3f_M=%d.txt' % (w_close,M_w_opt),a_w)

#plt.show()
