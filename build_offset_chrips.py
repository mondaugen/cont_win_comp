# Build many chirps but offset their frequency coefficient so that their
# maxmimum bin is bin 0
 
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

rslt=[]
for n in xrange(N_chirps):
    a0=np.random.uniform(a0_r[0],a0_r[1])
    a1=np.random.uniform(a1_r[0],a1_r[1])
    a2=np.random.uniform(a2_r[0],a2_r[1])
    b0=np.random.uniform(b0_r[0],b0_r[1])
    b1=np.random.uniform(b1_r[0],b1_r[1])
    b2=np.random.uniform(b2_r[0],b2_r[1])
    chirp=ddm.chirp_p2(a0,a1,a2,b0,b1,b2)
    kma0=np.argmax(np.abs(chirp.X(w)))
    chirp_new=ddm.chirp_p2(a0,a1,a2,b0,b1-2.*np.pi*kma0*float(F_s)/chirp._l,b2)
    kma0_new=np.argmax(np.abs(chirp_new.X(w)))
    rslt.append(kma0_new)
    #print kma0
    #print kma0_new
    assert(kma0_new==0)

with open('/tmp/rslt','w') as f:
    pickler=pickle.Pickler(f)
    pickler.dump(rslt)
