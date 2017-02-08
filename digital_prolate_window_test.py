import numpy as np
import numpy.linalg
import ddm
import matplotlib.pyplot as plt

# Length of window
N=25
#N=513
# Relative peak sidelobe height dB
a=-57.
# Mainlobe width in cycles/sample
b=(-.07401*a+1.007)/N
# Design coefficient for -49 > a
W=(-3.81e-2*a+3.16e-1)/N
#W=.8
i=np.arange(N,dtype='double')
A=np.diag(np.cos(2*np.pi*W)*(((N-1.)/2.-i)**2.))
A+=np.diag(0.5*i[1:]*(N-i[1:]),-1)
A+=np.diag(0.5*i[1:]*(N-i[1:]),1)
#(l,v)=ddm.eig_pow_method(A)
(L,V)=np.linalg.eig(A)
v=V[:,np.argmax(np.abs(L))]
fig,(ax1,ax2)=plt.subplots(2,1)
ax1.plot(i,np.real(v))
ax2.plot(np.arange(512),20.*np.log10(np.abs(np.fft.fft(v[:-1],512))))
plt.show()
