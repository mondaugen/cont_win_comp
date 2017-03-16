import ddm
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm

import warnings

plt.rc('text',usetex=True)
plt.rc('font',**ddm.FONT_OPT['dafx'])

show_plots=False

#warnings.simplefilter('error','RuntimeWarning')

F_s=16000
W_l_ms=32
R=3
# prepare chirps
N_chirps=10

chirps=ddm.build_offset_chirps(N_chirps=N_chirps,
                               # Sample rate (Hz)
                               F_s=F_s,
                               # Window length (ms)
                               W_l_ms=W_l_ms,
                               b1_r=[0.,np.pi*F_s])

wins=['hann','c1-nuttall-4','c1-nuttall-3','prolate-0.008-approx-5']
line_stys=['solid','dotted','dashed','dashdot']
labels=['H','C4','C3','P5']
sr_min=-30
srs=np.arange(sr_min,10,30)
n_diffs=40
diffs=xrange(n_diffs)
errs=dict()
for w in wins:
    errs[w]=dict()
    for s in srs:
        errs[w][s]=dict()
        errs[w][s]['diffs']=[0 for _ in diffs]
        errs[w][s]['diffs_log']=[0 for _ in diffs]
        errs[w][s]['diffs_params']=dict()
        errs[w][s]['diffs_params']['a0']=[0 for _ in diffs]
        errs[w][s]['diffs_params']['a1']=[0 for _ in diffs]
        errs[w][s]['diffs_params']['a2']=[0 for _ in diffs]
        errs[w][s]['diffs_params']['b0']=[0 for _ in diffs]
        errs[w][s]['diffs_params']['b1']=[0 for _ in diffs]
        errs[w][s]['diffs_params']['b2']=[0 for _ in diffs]

cmap=plt.get_cmap('Greys')

# Chirp creator
cp2c=ddm.chirp_p2_creator(W_l_ms,F_s)

# For all windows
for w in wins:
    # For all desired signal ratios
    for s in srs:
        # for all differences
        for d in diffs:
            #iterate through all pairs of chirps
            nlsqerrs=0
            errs[w][s]['diffs'][d]=0.
            errs[w][s]['diffs_log'][d]=0.
            for c1 in chirps:
                for c2 in list(set(chirps) - set([c1])):
                    p_x1=c1.p_x()
                    p_x2=c2.p_x()
                    # desired power ratio
                    p_=10.**(s/10.)*p_x1
                    q=np.log(p_/p_x2)/2.
                    c2_=cp2c.create_chirp(c2.a0 + q,
                                          c2.a1,
                                          c2.a2,
                                          c2.b0,
                                          # scaling this should have no influence on
                                          # the power, but should move up the bin
                                          # index
                                          c2.b1 + 2.*np.pi*d*float(F_s)/cp2c.l,
                                          c2.b2)
                    # Sum to make final signal
                    x=c1.x+c2_.x
                    try:
                        #errs_1=ddm.p2_1_3_est(x,c1,0,w,F_s)
                        #errs_2=ddm.p2_1_3_est(x,c2_,d,w,F_s)
                        errs_1=ddm.p2_1_3_est(x,c1,0,w,F_s,R)
                        errs_2=ddm.p2_1_3_est(x,c2_,d,w,F_s,R)
                    except TypeError:
                        nlsqerrs += 1
                        continue
                    errs[w][s]['diffs'][d] += (np.sum(np.power(errs_1,2.)) +
                                                np.sum(np.power(errs_2,2.)))
                    for param,err1,err2 in zip(['a0','a1','a2','b0','b1','b2'],
                                            errs_1,errs_2):
                        errs[w][s]['diffs_params'][param][d] =  (err1**2. +
                                                                    err2**2.)

            for param,err1,err2 in zip(['a0','a1','a2','b0','b1','b2'],
                                    errs_1,errs_2):
                errs[w][s]['diffs_params'][param][d] /= len(chirps)*(len(chirps)-1) - nlsqerrs
            errs[w][s]['diffs'][d] /= len(chirps)*(len(chirps)-1) - nlsqerrs
            errs[w][s]['diffs_log'][d] =np.log10(errs[w][s]['diffs'][d])
            #print 'nerrs: %d for sr %f' % (nlsqerrs,s)

fig=plt.figure(1)
fig2,axs=plt.subplots(3,2,sharex=True)

diffs_min=1000.
diffs_max=-1000.
for w,ls,lab in zip(wins,line_stys,labels):
    for s in srs:
        clr=cmap((-1.*s+20.)/(-1.*float(sr_min)+20.))
        #print errs[w][s]['diffs_log']
        plt.figure(1)
        plt.plot(diffs,
                errs[w][s]['diffs_log'],c=clr,ls=ls,label=lab + ' $%d$ dB' % (s,))
        if np.max(errs[w][s]['diffs_log']) > diffs_max:
            diffs_max = np.max(errs[w][s]['diffs_log'])
        if np.min(errs[w][s]['diffs_log']) < diffs_min:
            diffs_min = np.min(errs[w][s]['diffs_log'])
        plt.figure(2)
        for plt_i,param_name in enumerate(['a0','a1','a2']):
            axs[plt_i,0].plot(diffs,
                np.log10(errs[w][s]['diffs_params'][param_name]),c=clr,ls=ls,label=lab + ' $%d$ dB' % (s,))
        for plt_i,param_name in enumerate(['b0','b1','b2']):
            axs[plt_i,1].plot(diffs,
                np.log10(errs[w][s]['diffs_params'][param_name]),c=clr,ls=ls,label=lab + ' $%d$ dB' % (s,))

phs_poly_param_items=[('a_{0,0}','a_{0,0}','a_{1,0}','a_{1,0}'),('a_{0,1}','a_{0,1}','a_{1,1}','a_{1,1}'),('a_{0,2}','a_{0,2}','a_{1,2}','a_{1,2}')]
re_param_names=['mean$\{(\Re\{\hat{%s}\}-\Re\{%s\})^2+(\Re\{\hat{%s}\}-\Re\{%s\})^2\}$' % s_ for s_ in phs_poly_param_items]
im_param_names= ['mean$\{(\Im\{\hat{%s}\}-\Im\{%s\})^2+(\Im\{\hat{%s}\}-\Im\{%s\})^2\}$' % s_ for s_ in phs_poly_param_items]
for plt_i,param_name in enumerate(re_param_names):
    axs[plt_i,0].set_title(param_name,fontsize=10)
    axs[plt_i,0].set_ylabel('$\log_{10}$ MSE')
    axs[plt_i,0].set_xlim([0,n_diffs-1])

for plt_i,param_name in enumerate(im_param_names):
    axs[plt_i,1].set_title(param_name,fontsize=10)
    axs[plt_i,1].set_xlim([0,n_diffs-1])

axs[2,0].set_xlabel('Difference between maxima in bins')
axs[2,1].set_xlabel('Difference between maxima in bins')

axs[2,1].legend(fontsize=10)
fig2.suptitle(
    'Average parameter estimation error variance for mixture of 2 components',
    # hacky way to get title size
    fontsize=fig.get_axes()[0].title.get_fontproperties().get_size_in_points())


#fig2.tight_layout()
plt.subplots_adjust(top=.92)
fig2.set_size_inches(10,9)


plt.figure(1)
plt.xlabel('Difference between maxima in bins')
plt.ylabel('log10 MSE')
plt.legend(fontsize=10,loc='upper right')
plt.xlim([0,n_diffs-1])
plt.ylim([diffs_min,diffs_max])
plt.title('Total mean-squared estimation error for two signals at various ' +
    'signal power ratios')

plt.savefig('paper/comp_offset_chirp_est_err.eps')
fig2.savefig('paper/comp_offset_chirp_est_err_params.eps')
with open('paper/comp_offset_chirp_est_err_defs.txt','w') as f:
    f.write('\\newcommand{\Koffset}{%d}\n' % (N_chirps,))
    f.write('\\newcommand{\Doffset}{%d}\n' % (n_diffs,))

if (show_plots):
    plt.show()
