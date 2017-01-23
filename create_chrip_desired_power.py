# Create a chirp and then adjust its first real coefficient until it is the
# correct power

import ddm
import numpy as np
import pickle

N_chirps=1000

# Sample rate (Hz)
F_s=16000
# Window length (ms)
W_l_ms=32
# Window length (samples)
W_l=F_s*(W_l_ms/1000.)
# Analysis window, and derivative
w,dw=ddm.w_dw_sum_cos(W_l,'hann',norm=False)

# Chirp creator
cp2c=ddm.chirp_p2_creator(W_l_ms,F_s)

# Parameter ranges
# Amplitude part
a0_r=[0,1.]
a1_r=[-100.,100.]
a2_r=[-1.e4,1.e3]
a3_r=[-1.e5,1.e5]
# Phase part
b0_r=[-np.pi,np.pi]
b1_r=[0.,np.pi*F_s]
#b1_r=[0.,0.]
b2_r=[-1.e4,1.e4]
#b2_r=[1.e4,1.e4]
b3_r=[-1.e6,1.e6]

chirp=cp2c.create_random_chirp(a0_r,a1_r,a2_r,b0_r,b1_r,b2_r)
p_x=chirp.p_x()
print "original power %f" % (p_x,)
# desired power is 1
q=np.log(1./p_x)/2.
print "q %f" % (q,)
chirp_new=cp2c.create_chirp(chirp.a0 + q,
                            chirp.a1,
                            chirp.a2,
                            chirp.b0,
                            chirp.b1, # scaling this should have no influence
                            chirp.b2)
p_x_new=chirp_new.p_x()
print "new power %f" % (p_x_new,)
