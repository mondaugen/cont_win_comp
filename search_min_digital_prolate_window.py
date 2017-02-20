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
W=np.arange(.01,.99,.01)
# Ratio of max to peak sidelobe (dB)
a=np.zeros(len(W))
# The ratio desired
a_des=93.32
#W=.8
i=np.arange(N,dtype='double')

w_min=None
a_min=0
v_min=None
v_F_min=None
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
    if (a[n] > a_min):
        a_min=a[n]
        w_min=W_
        v_min=v
        v_F_min=v_F

print "a_min: %f" % (a_min,)
print "v_min end point: %g" % (v_min[0],)
print "w_min: %f" % (w_min,)

fig, (ax1,ax2) = plt.subplots(2,1)
ax1.plot(W,a)
ax1.set_title("W versus peak side-lobe height")
w_nuttall=ddm.w_dw_sum_cos(N-1,a='c1-nuttall-4')[0]
w_F_nuttall=20.*np.log10(np.abs(np.fft.fft(w_nuttall,N_F)))
w_F_nuttall=np.concatenate((w_F_nuttall[N_F/2:],w_F_nuttall[:N_F/2]))
leg_nuttall,=ax2.plot(np.arange(N_F),w_F_nuttall-np.max(w_F_nuttall),c='r',label="Nuttall")
leg_prolate,=ax2.plot(np.arange(N_F),v_F_min-np.max(v_F_min),c='b',label="Prolate")
ax2.set_title("nuttall versus prolate")
ax2.legend(handles=[leg_nuttall,leg_prolate])
print "w_nuttall end point: %g" % (w_nuttall[0],)
plt.show()
