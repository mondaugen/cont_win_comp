# The distribution derivative method and windows it requires

import numpy as np
import matplotlib.pyplot as plt

def w_dw_sum_cos(M,a='hanning',norm=False):
    """
    Construct sum of cosine windows, based on type or coefficients specified in
    a.

    M:
        Integer. Length of window.
    a:
        str or iterable. If string, can be:
            'hann'
            'hanning'
            'c1-blackman-4'
        Otherwise can specify the coefficients of a window.
    norm:
        If norm is true, multiply by constant so that ||w||^2 is 1

    Returns:
    w,dw
    w:
        The window.
    dw:
        The derivative of window
    """
    
    if (type(a)==type(str())):
        if (a=='hann') or (a=='hanning'):
            c=np.r_[0.5,0.5]
        elif (a=='c1-blackman-4'):
            c=np.r_[0.358735,0.488305,0.141265,0.011695]
        elif (a=='c1-nuttall-3'):
            c=np.r_[0.40897,0.5,0.09103]
        elif (a=='c1-nuttall-4'):
            c=np.r_[0.355768,0.487396,0.144232,0.012604]
    else:
        try:
            _it=iter(a)
        except TypeError:
            print a, 'is not iterable'
        c=a
    m=np.arange(M)
    w_=((c*np.power(-1,np.arange(len(c))))[:,np.newaxis]
            *np.cos(np.pi*2./M*np.outer(np.arange(len(c)),m)))
    w=np.sum(w_,0)
    C=1
    if (norm):
        C=np.sum(w)
    w/=C

    dw_=((2.*np.pi/M*c[1:]*np.arange(1,len(c))
            *np.power(-1,1+np.arange(1,len(c))))[:,np.newaxis]
        *np.sin(np.pi*2./M*np.outer(np.arange(1,len(c)),m)))
    dw=np.sum(dw_,0)
    dw/=C
    return (w,dw)

def ddm_p2_1_3(x,w,dw,kma0):
    """
    Compute parameters of 2nd order polynomial using 2 bins surrounding the
    maximum of the STFT at the centre of signal x.

    x:  
        the signal to analyse, must have length of at least N_w where N_w is
        the length of the window.
    w:  
        the analysis window.
    dw: 
        the derivative of the analysis window.
    kma0:
        the index of the maximum. From here the atoms are chosen as the one
        corresponding to this index and the two adjacent ones.

    Returns 

    a:
        a vector containing the estimated parameters.
    """
    N_w=len(w)
    nx0=np.arange(N_w)
    x0=x[nx0]
    Xp1w=np.fft.fft(x0*w)
    Xp2w=np.fft.fft(2.*nx0*x0*w)
    Xdw_=np.fft.fft(x0*dw)
    Xdw=Xp1w*(-2.*np.pi*1j*nx0/N_w)+Xdw_
    result=[]
    if (kma0 == 0) or (kma0 == (N_w-1)):
        return None
    kma0__1=(kma0-1)%N_w
    kma0_1=(kma0+1)%N_w
    A=np.c_[
            np.r_[
                Xp1w[(kma0)-1:(kma0+2)],
                ],
            np.r_[
                Xp2w[(kma0)-1:(kma0+2)],
                ]
            ]
    c=np.c_[
            np.r_[
                Xdw[(kma0)-1:(kma0+2)],
                ]
            ]
    try:
        a=np.linalg.lstsq(A,-c)[0]
        gam=np.exp(a[0]*nx0+a[1]*nx0**2.)
        a0=(np.log(np.inner(x0,np.conj(gam)))
            -np.log(np.inner(gam,np.conj(gam))))
#        a0=(np.log(Xp1w[kma0])-np.log(np.fft.fft(gam*w)[kma0]))
        result.append(np.vstack((a0,a)))
    except ValueError:
        return None
    return result

def plot_sig_dft_c(x,show=True):
    """
    Plot the DFT of a signal so that frequency 0 is in the middle of the plot.
    """
    if (len(x) % 2) != 0:
        raise ValueError('Length must be even')
    X=np.fft.fft(x)
    n=np.arange(len(x))
    X=np.r_[X[len(X)/2:],X[:len(X)/2]]
    n=np.r_[n[len(n)/2:]-len(n),n[:len(n)/2]]
    plt.plot(n,20*np.log10(np.abs(X)))
    if (show):
        plt.show()
    return (n,X)

class chirp_p2:
    def __init__(self,a0=0,a1=0,a2=0,b0=0,b1=0,b2=0,l=16000*(32./1000.),Fs=16000):
        self.a0=a0
        self.a1=a1
        self.a2=a2
        self.b0=b0
        self.b1=b1
        self.b2=b2
        t=np.arange(l)/float(Fs)
        arg_re=np.polyval([self.a2,self.a1,self.a0],t)
        arg_im=np.polyval([self.b2,self.b1,self.b0],t)
        self.x=np.exp(arg_re+1j*arg_im)
    def X(self,w):
        """
        Get the frequency domain representation of the chirp by performing the
        DFT.

        w:
            a window to window the signal with before transforming.
        """
        if not (len(w) == len(self.x)):
            raise ValueError('Window must be same length as signal.')
        return np.fft.fft(self.x*w)
    def p_x(self):
        # Power of analytic signal
        return np.sum(self.x**2.)/float(len(self.x))

class chirp_p2_creator:
    def __init__(self,l_ms=32,Fs=16000):
        self.l_ms=32
        self.Fs=16000
        self.l=Fs*(l_ms/1000.)
    def create_chirp(self,a0=0,a1=0,a2=0,b0=0,b1=0,b2=0):
        return chirp_p2(a0,a1,a2,b0,b1,b2,self.l,self.Fs)
    def create_random_chirp(self,a0_r,a1_r,a2_r,b0_r,b1_r,b2_r):
        a0=np.random.uniform(a0_r[0],a0_r[1])
        a1=np.random.uniform(a1_r[0],a1_r[1])
        a2=np.random.uniform(a2_r[0],a2_r[1])
        b0=np.random.uniform(b0_r[0],b0_r[1])
        b1=np.random.uniform(b1_r[0],b1_r[1])
        b2=np.random.uniform(b2_r[0],b2_r[1])
        return create_chirp(a0,a1,a2,b0,b1,b2)
