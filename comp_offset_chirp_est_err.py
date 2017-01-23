import ddm
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm

import warnings

#warnings.simplefilter('error','RuntimeWarning')

F_s=16000
W_l_ms=32

# prepare chirps

chirps=ddm.build_offset_chirps(N_chirps=2,
                               # Sample rate (Hz)
                               F_s=F_s,
                               # Window length (ms)
                               W_l_ms=W_l_ms,
                               b1_r=[0.,np.pi*F_s])

wins=['hann','c1-nuttall-4','c1-blackman-4']
line_stys=['solid','dotted','dashed']
sr_min=-60
srs=np.arange(sr_min,0,10)
n_diffs=64
diffs=xrange(n_diffs)
errs=dict()
for w in wins:
    errs[w]=dict()
    for s in srs:
        errs[w][s]=dict()
        errs[w][s]['diffs']=[0 for _ in diffs]
        errs[w][s]['diffs_log']=[0 for _ in diffs]

cmap=plt.get_cmap('gnuplot')

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
                for c2 in chirps:
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
                        errs_1=ddm.p2_1_3_est(x,c1,0,w,F_s)
                        errs_2=ddm.p2_1_3_est(x,c2_,d,w,F_s)
                    except TypeError:
                        nlsqerrs += 1
                        continue
                    errs[w][s]['diffs'][d] += np.sum(errs_1) + np.sum(errs_2)
            errs[w][s]['diffs'][d] /= len(chirps)**2. - nlsqerrs
            errs[w][s]['diffs_log'][d] =np.log10(errs[w][s]['diffs'][d])
            print 'nerrs: %d for sr %f' % (nlsqerrs,s)

fig=plt.figure(1)

for w,ls in zip(wins,line_stys):
    for s in srs:
        clr=cmap(s/float(sr_min))
        print errs[w][s]['diffs_log']
        plt.plot(diffs,
                errs[w][s]['diffs_log'],c=clr,ls=ls)

plt.xlabel('Difference of between maxima in bins')
plt.ylabel('log10 MSE')

plt.show()
