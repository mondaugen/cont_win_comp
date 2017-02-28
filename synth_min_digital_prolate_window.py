import numpy as np
import numpy.linalg
import ddm
import matplotlib.pyplot as plt
import sigmod as sm

plt.rc('text',usetex=True)
plt.rc('font',family='serif')

# Length of window
N=513
# Length of DFT (controls zero-padding)
N_F=1024
#N=513
# Design coefficients
# This coefficient found to give lowest side lobes
W=0.77
v=ddm.psw_design(N,W)
# If negative, invert (will not affect maginitude reponse)
if (np.min(v) < -0.1):
    v*=-1.
# Make maximum 1.
v/=np.max(v)
v_F=20.*np.log10(np.abs(np.fft.fft(v,N_F)))
v_F=np.concatenate((v_F[N_F/2:],v_F[:N_F/2]))
w_min=W
v_min=v
v_F_min=v_F

print "v_min end point: %g" % (v_min[0],)
print "w_min: %f" % (w_min,)

fig, (ax1,ax2) = plt.subplots(2,1)
ax1.plot(np.arange(N),v)
ax1.set_title("Spheroidal window, time-domain")
w_nuttall=ddm.w_dw_sum_cos(N-1,a='c1-nuttall-4')[0]
w_F_nuttall=20.*np.log10(np.abs(np.fft.fft(w_nuttall,N_F)))
w_F_nuttall=np.concatenate((w_F_nuttall[N_F/2:],w_F_nuttall[:N_F/2]))
leg_nuttall,=ax2.plot(np.arange(N_F),w_F_nuttall-np.max(w_F_nuttall),c='r',label="Nuttall")
leg_prolate,=ax2.plot(np.arange(N_F),v_F_min-np.max(v_F_min),c='b',label="Prolate")
ax2.set_title("nuttall versus prolate, frequency domain")
ax2.legend(handles=[leg_nuttall,leg_prolate])
print "w_nuttall end point: %g" % (w_nuttall[0],)
plt.show()
