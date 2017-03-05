import numpy as np
import numpy.linalg
import ddm
import matplotlib.pyplot as plt
import sigmod as sm

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
bw_des=2.
#bw_des=1.91
#W=.8
i=np.arange(N,dtype='double')
# Number of cosines to use in approximation
M_w=N/8
# Constraints to use when approximating (just continuous)
c_w={'c1'}

w_close=None
bw_diff_close=100.
v_close=None
v_F_close=None
for n,W_ in enumerate(W):
    #A=np.diag(np.cos(2*np.pi*W_)*(((N-1.)/2.-i)**2.))
    #A+=np.diag(0.5*i[1:]*(N-i[1:]),-1)
    #A+=np.diag(0.5*i[1:]*(N-i[1:]),1)
    ##(l,v)=ddm.eig_pow_method(A)
    #(L,V)=np.linalg.eig(A)
    #v=V[:,np.argmax(np.abs(L))]
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

v_F_close=np.concatenate((v_F_close[N_F/2:],v_F_close[:N_F/2]))
a=[None if a_ > 100 else a_ for a_ in a]
i_a_valid=[i for (i,val) in enumerate(a) if val != None]
a=[a[i] for i in i_a_valid]
W=[W[i] for i in i_a_valid]

print "bw_diff_close: %f" % (bw_diff_close,)
print "v_close end point: %g" % (v_close[0],)
print "w_close: %g" % (w_close,)

# Approximate designed window using sum of cosines
a_w=ddm.cos_approx_win(v,M_w,c_w)
w_aprx=ddm.w_dw_sum_cos(N-1,a=a_w)[0]
w_aprx_F=20.*np.log10(np.abs(np.fft.fft(w_aprx,N_F)))
# Put window centre in middle of plot
w_aprx_F=np.concatenate((w_aprx_F[N_F/2:],w_aprx_F[:N_F/2]))

# Plot windows
fig, (ax1,ax2) = plt.subplots(2,1)
ax1.plot(W,a)
ax1.set_title("W versus 3db bandwidth")

w_nuttall=ddm.w_dw_sum_cos(N-1,a='c1-nuttall-4')[0]
w_F_nuttall=20.*np.log10(np.abs(np.fft.fft(w_nuttall,N_F)))
w_F_nuttall=np.concatenate((w_F_nuttall[N_F/2:],w_F_nuttall[:N_F/2]))
leg_nuttall,=ax2.plot(np.arange(N_F)/(float(N_F)/(N-1)),
        w_F_nuttall-np.max(w_F_nuttall),c='r',label="Nuttall")
leg_prolate,=ax2.plot(np.arange(N_F)/(float(N_F)/(N-1)),
        v_F_close-np.max(v_F_close),c='b',label="Prolate")
leg_aprx,=ax2.plot(np.arange(N_F)/(float(N_F)/(N-1)),
        w_aprx_F-np.max(w_aprx_F),c='g',label="Approx. Prolate")
ax2.set_title("nuttall versus prolate and its approximation")
ax2.legend(handles=[leg_nuttall,leg_prolate,leg_aprx])
print "w_nuttall end point: %g" % (w_nuttall[0],)
plt.show()
