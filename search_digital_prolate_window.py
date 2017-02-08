import numpy as np
import numpy.linalg
import ddm
import matplotlib.pyplot as plt
import sigmod as sm

# Length of window
N=513
# Length of DFT (controls zero-padding)
N_F=1024
#N=513
# Design coefficients
# This range found empirically to give peak side-lobes close to desired one
W=np.arange(.005,.01,0.0001)
# Ratio of max to peak sidelobe (dB)
a=np.zeros(len(W))
# The ratio desired
a_des=93.32
#W=.8
i=np.arange(N,dtype='double')

w_close=None
a_diff_close=100.
v_close=None
v_F_close=None
for n,W_ in enumerate(W):
    A=np.diag(np.cos(2*np.pi*W_)*(((N-1.)/2.-i)**2.))
    A+=np.diag(0.5*i[1:]*(N-i[1:]),-1)
    A+=np.diag(0.5*i[1:]*(N-i[1:]),1)
    #(l,v)=ddm.eig_pow_method(A)
    (L,V)=np.linalg.eig(A)
    v=V[:,np.argmax(np.abs(L))]
    v_F=20.*np.log10(np.abs(np.fft.fft(v,N_F)))
    v_F=np.concatenate((v_F[N_F/2:],v_F[:N_F/2]))
    (ma,mai)=sm.lextrem(v_F)
    ma_srt=np.sort(ma.flatten())
    a[n]=ma_srt[-1]-ma_srt[-2]
    a_diff=np.abs(a_des-a[n])
    if (a_diff < a_diff_close):
        a_diff_close=a_diff
        w_close=W_
        v_close=v
        v_F_close=v_F

print "a_diff_close: %f" % (a_diff_close,)
print "v_close end point: %g" % (v_close[0],)

fig, (ax1,ax2) = plt.subplots(2,1)
ax1.plot(W,a)
ax1.set_title("W versus peak side-lobe height")
w_nuttall=ddm.w_dw_sum_cos(N-1,a='c1-nuttall-4')[0]
w_F_nuttall=20.*np.log10(np.abs(np.fft.fft(w_nuttall,N_F)))
w_F_nuttall=np.concatenate((w_F_nuttall[N_F/2:],w_F_nuttall[:N_F/2]))
leg_nuttall,=ax2.plot(np.arange(N_F),w_F_nuttall-np.max(w_F_nuttall),c='r',label="Nuttall")
leg_prolate,=ax2.plot(np.arange(N_F),v_F_close-np.max(v_F_close),c='b',label="Prolate")
ax2.set_title("nuttall versus prolate")
ax2.legend(handles=[leg_nuttall,leg_prolate])
print "w_nuttall end point: %g" % (w_nuttall[0],)
plt.show()
