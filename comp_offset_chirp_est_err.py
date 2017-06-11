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
localmax=False

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
labels=['H','N4','N3','P5']
sr_min=-30
srs=np.arange(sr_min,10,30)

# make colours
cmap=plt.get_cmap('magma')
clrs=[(x+1.)/float(len(srs)+1.) for x in xrange(len(srs))]
clr_dict=dict()
for clr, s in zip(clrs,srs):
    clr_dict[s]=cmap(clr)

n_diffs=40
diff_step=0.25
diffs=np.arange(0,n_diffs,diff_step)
errs=dict()
for w in wins:
    errs[w]=dict()
    for s in srs:
        errs[w][s]=dict()
        errs[w][s]['diffs']=[0 for _ in diffs]
        errs[w][s]['diffs_sig']=[0 for _ in diffs]
        errs[w][s]['diffs_log']=[0 for _ in diffs]
        errs[w][s]['diffs_sig_log']=[0 for _ in diffs]
        errs[w][s]['diffs_params']=dict()
        errs[w][s]['diffs_params']['a0']=[0 for _ in diffs]
        errs[w][s]['diffs_params']['a1']=[0 for _ in diffs]
        errs[w][s]['diffs_params']['a2']=[0 for _ in diffs]
        errs[w][s]['diffs_params']['b0']=[0 for _ in diffs]
        errs[w][s]['diffs_params']['b1']=[0 for _ in diffs]
        errs[w][s]['diffs_params']['b2']=[0 for _ in diffs]

# Chirp creator
cp2c=ddm.chirp_p2_creator(W_l_ms,F_s)

# For all windows
for w in wins:
    # For all desired signal ratios
    for s in srs:
        # for all differences
        for di,d in enumerate(diffs):
            #iterate through all pairs of chirps
            nlsqerrs=0
            errs[w][s]['diffs'][di]=0.
            errs[w][s]['diffs_log'][di]=0.
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
                        errs_1,params_1=ddm.p2_1_3_est(x,c1,0,w,F_s,R)
                        errs_2,params_2=ddm.p2_1_3_est(x,c2_,d,w,F_s,R)
                    except TypeError:
                        nlsqerrs += 1
                        continue
                    errs[w][s]['diffs'][di] += (np.sum(np.power(errs_1,2.)) +
                                                np.sum(np.power(errs_2,2.)))
                    for param,err1,err2 in zip(['a0','a1','a2','b0','b1','b2'],
                                            errs_1,errs_2):
                        errs[w][s]['diffs_params'][param][di] =  (err1**2. +
                                                                    err2**2.)
                    # Synthesize chirp combo from estimated parameters and
                    # evaluate error
                    #x_=cp2c.create_chirp(*params_1).x+cp2c.create_chirp(*params_2).x
                    #errs[w][s]['diffs_sig'][di] += np.sum(
                    #        np.real((x_-x)*np.conj(x_-x)))/len(x)
                    x_err=c2_.x+c1.x-cp2c.create_chirp(*params_2).x-cp2c.create_chirp(*params_1).x
                    errs[w][s]['diffs_sig'][di] += np.sum(
                            np.real((x_err)*np.conj(x_err)))/len(x_err)

            for param,err1,err2 in zip(['a0','a1','a2','b0','b1','b2'],
                                    errs_1,errs_2):
                errs[w][s]['diffs_params'][param][di] /= len(chirps)*(len(chirps)-1) - nlsqerrs
            errs[w][s]['diffs'][di] /= len(chirps)*(len(chirps)-1) - nlsqerrs
            errs[w][s]['diffs_log'][di] =np.log10(errs[w][s]['diffs'][di])
            errs[w][s]['diffs_sig'][di] /= len(chirps)*(len(chirps)-1) - nlsqerrs
            errs[w][s]['diffs_sig_log'][di] =np.log10(errs[w][s]['diffs_sig'][di])
            #print 'nerrs: %d for sr %f' % (nlsqerrs,s)

fig=plt.figure(1)
fig2,axs=plt.subplots(3,2,sharex=True)
fig3,axs3=plt.subplots(1,1)

diffs_min=1000.
diffs_max=-1000.
diffs_sig_min=1000.
diffs_sig_max=-1000.
diffs_min2=1000.
diffs_max2=-1000.
diff_extrem_margin=0.5
for w,ls,lab in zip(wins,line_stys,labels):
    for s in srs:
        clr=clr_dict[s]
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
            if (localmax):
                plotme,msk=ddm.localmax(errs[w][s]['diffs_params'][param_name])
            else:
                plotme,msk=(errs[w][s]['diffs_params'][param_name],
                        np.arange(len(errs[w][s]['diffs_params'][param_name])))
            axs[plt_i,0].plot(diffs[msk],
                np.log10(plotme),c=clr,ls=ls,label=lab + ' $%d$ dB' % (s,))
            if np.max(plotme) > diffs_max2:
                diffs_max2 =np.max(plotme) 
            if np.min(plotme) < diffs_min2:
                diffs_min2 = np.min(plotme)
        for plt_i,param_name in enumerate(['b0','b1','b2']):
            if (localmax):
                plotme,msk=ddm.localmax(errs[w][s]['diffs_params'][param_name])
            else:
                plotme,msk=(errs[w][s]['diffs_params'][param_name],
                        np.arange(len(errs[w][s]['diffs_params'][param_name])))
            axs[plt_i,1].plot(diffs[msk],
                np.log10(plotme),c=clr,ls=ls,label=lab + ' $%d$ dB' % (s,))
            if np.max(plotme) > diffs_max2:
                diffs_max2 =np.max(plotme) 
            if np.min(plotme) < diffs_min2:
                diffs_min2 = np.min(plotme)
        axs3.plot(diffs,
            errs[w][s]['diffs_sig_log'],c=clr,ls=ls,label=lab + ' $%d$ dB' % (s,))
        if np.max(errs[w][s]['diffs_sig_log']) > diffs_sig_max:
            diffs_sig_max = np.max(errs[w][s]['diffs_sig_log'])
        if np.min(errs[w][s]['diffs_sig_log']) < diffs_sig_min:
            diffs_sig_min = np.min(errs[w][s]['diffs_sig_log'])


phs_poly_param_items=[('a_{0,0}','a_{0,0}','a_{1,0}','a_{1,0}'),('a_{0,1}','a_{0,1}','a_{1,1}','a_{1,1}'),('a_{0,2}','a_{0,2}','a_{1,2}','a_{1,2}')]
re_param_names=['mean$\{(\Re\{\hat{%s}\}-\Re\{%s\})^2+(\Re\{\hat{%s}\}-\Re\{%s\})^2\}$' % s_ for s_ in phs_poly_param_items]
im_param_names= ['mean$\{(\Im\{\hat{%s}\}-\Im\{%s\})^2+(\Im\{\hat{%s}\}-\Im\{%s\})^2\}$' % s_ for s_ in phs_poly_param_items]
for plt_i,param_name in enumerate(re_param_names):
    axs[plt_i,0].set_title(param_name,fontsize=10)
    axs[plt_i,0].set_ylabel('$\log_{10}$ MSE')
    axs[plt_i,0].set_xlim([0,n_diffs-1])
    axs[plt_i,0].set_ylim([np.log10(diffs_min2)-diff_extrem_margin,np.log10(diffs_max2)+diff_extrem_margin])


for plt_i,param_name in enumerate(im_param_names):
    axs[plt_i,1].set_title(param_name,fontsize=10)
    axs[plt_i,1].set_xlim([0,n_diffs-1])
    axs[plt_i,1].set_ylim([np.log10(diffs_min2)-diff_extrem_margin,np.log10(diffs_max2)+diff_extrem_margin])

axs[2,0].set_xlabel('Difference between maxima in bins')
axs[2,1].set_xlabel('Difference between maxima in bins')

axs[0,1].legend(handlelength=3.0,fontsize=10)
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

axs3.set_xlabel('Difference between maxima in bins')
axs3.set_ylabel('log10 MSE')
axs3.legend(fontsize=10,loc='upper right')
axs3.set_xlim([0,n_diffs-1])
axs3.set_ylim([diffs_sig_min,diffs_sig_max])
axs3.set_title('Total mean-squared estimation error for two signals at various ' +
    'signal power ratios')

plt.savefig('paper/comp_offset_chirp_est_err.eps')
fig2.savefig('paper/comp_offset_chirp_est_err_params.eps')
fig3.savefig('paper/comp_offset_chirp_est_err_sigs.eps')
with open('paper/comp_offset_chirp_est_err_defs.txt','w') as f:
    f.write('\\newcommand{\Koffset}{%d}\n' % (N_chirps,))
    f.write('\\newcommand{\Doffset}{%d}\n' % (n_diffs,))
    f.write('\\newcommand{\Dstep}{%.2f}\n' % (diff_step,))

if (show_plots):
    plt.show()
