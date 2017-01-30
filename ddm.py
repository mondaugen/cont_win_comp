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
    #kma0__1=float((kma0-1)%N_w)
    #kma0_1=float((kma0+1)%N_w)
    kma0__1=float((kma0-1))
    kma0_1=float((kma0+1))
    kma0=float(kma0)
    Xp1w=np.r_[
            np.sum(w*x0*np.exp(-2.*np.pi*1j*nx0*kma0__1/N_w)),
            np.sum(w*x0*np.exp(-2.*np.pi*1j*nx0*kma0/N_w)),
            np.sum(w*x0*np.exp(-2.*np.pi*1j*nx0*kma0_1/N_w))
            ]
    Xp2w=np.r_[
            np.sum(w*2.*nx0*x0*np.exp(-2.*np.pi*1j*nx0*kma0__1/N_w)),
            np.sum(w*2.*nx0*x0*np.exp(-2.*np.pi*1j*nx0*kma0/N_w)),
            np.sum(w*2.*nx0*x0*np.exp(-2.*np.pi*1j*nx0*kma0_1/N_w))
            ]
    Xdw_=np.r_[
            np.sum(dw*x0*np.exp(-2.*np.pi*1j*nx0*kma0__1/N_w)),
            np.sum(dw*x0*np.exp(-2.*np.pi*1j*nx0*kma0/N_w)),
            np.sum(dw*x0*np.exp(-2.*np.pi*1j*nx0*kma0_1/N_w))
            ]
    #Xp1w=np.fft.fft(x0*w)
    #Xp2w=np.fft.fft(2.*nx0*x0*w)
    #Xdw_=np.fft.fft(x0*dw)
#    Xdw=Xp1w*(-2.*np.pi*1j*nx0/N_w)+Xdw_
    Xdw=Xp1w*(-2.*np.pi*1j*np.r_[kma0__1,kma0,kma0_1]/float(N_w))+Xdw_
    result=[]
    #if (kma0 == 0) or (kma0 == (N_w-1)):
    #    return None
    A=np.c_[
#            np.r_[
##                Xp1w[(kma0)-1:(kma0+2)],
#                Xp1w[[kma0__1,kma0,kma0_1]],
#                ],
            Xp1w,
#            np.r_[
##                Xp2w[(kma0)-1:(kma0+2)],
#                Xp2w[[kma0__1,kma0,kma0_1]],
#                ]
            Xp2w
            ]
    c=np.c_[
            Xdw
#            np.r_[
##                Xdw[(kma0)-1:(kma0+2)]
#                Xdw[[kma0__1,kma0,kma0_1]],
#                ]
            ]
    try:
        a=np.linalg.lstsq(A,-c)[0]
        gam=np.exp(a[0]*nx0+a[1]*nx0**2.)
        t_=np.inner(x0,np.conj(gam))
        b_=np.inner(gam,np.conj(gam))
        a0=(np.log(t_)
            -np.log(b_))
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
        r = np.sum(self.x*np.conj(self.x))/float(len(self.x))
        return np.real(r)
    def p2_1_3_est(self,kma0,w_name='hann',F_s=16000):
        """
        Estimate the parameters of the chirp using the DDM and output the
        errors.

        kma0:
            The bin of the centre atom in the estimation scheme.
        w_name:
            The name of the kind of window to use (see: w_dw_sum_cos)
        """
        # Generate window
        w,dw=w_dw_sum_cos(len(self.x),w_name,norm=False)
        # Estimate parameters
        alpha=ddm_p2_1_3(self.x,w,dw,kma0)
        # DDM estimation is done with normalized frequency, so correct
        # coefficients
        a0_=np.real(alpha[0][0])
        a1_=np.real(alpha[0][1])*F_s
        a2_=np.real(alpha[0][2])*(F_s**2.)
        b0_=np.imag(alpha[0][0])
        while (b0_ > np.pi):
            b0_ -= 2.*np.pi
        while (b0_ < -np.pi):
            b0_ += 2.*np.pi
        b1_=np.imag(alpha[0][1])*F_s
        b2_=np.imag(alpha[0][2])*(F_s**2.)
        a0_err=np.abs(self.a0-a0_)
        a1_err=np.abs(self.a1-a1_)
        a2_err=np.abs(self.a2-a2_)
        # Find minimum distance on circle
        b0_err=min([(self.b0-b0_)%(2.*np.pi),(b0_-self.b0)%(2.*np.pi)])
        b1_err=np.abs(self.b1-b1_)
        b2_err=np.abs(self.b2-b2_)
        return (a0_err,a1_err,a2_err,b0_err,b1_err,b2_err)


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
        return self.create_chirp(a0,a1,a2,b0,b1,b2)

def build_offset_chirps(N_chirps=1000,
                        # Sample rate (Hz)
                        F_s=16000,
                        # Window length (ms)
                        W_l_ms=32,
                        # Parameter ranges
                        # Amplitude part
                        a0_r=[0,1.],
                        a1_r=[-100.,100.],
                        a2_r=[-1.e4,1.e3],
                        a3_r=[-1.e5,1.e5],
                        # Phase part
                        b0_r=[-np.pi,np.pi],
                        b1_r=[0.,np.pi*16000],
                        b2_r=[-1.e4,1.e4],
                        b3_r=[-1.e6,1.e6],
                        w_name='hann'):
    # Window length (samples)
    W_l=F_s*(W_l_ms/1000.)
    # Analysis window, and derivative
    w,dw=w_dw_sum_cos(W_l,w_name,norm=False)

    # Chirp creator
    cp2c=chirp_p2_creator(W_l_ms,F_s)

    rslt=[]
    for n in xrange(N_chirps):
        chirp=cp2c.create_random_chirp(a0_r,a1_r,a2_r,b0_r,b1_r,b2_r)
        kma0=np.argmax(np.abs(chirp.X(w)))
        chirp_new=cp2c.create_chirp(chirp.a0,
                                    chirp.a1,
                                    chirp.a2,
                                    chirp.b0,
                                    chirp.b1-2.*np.pi*kma0*float(F_s)/cp2c.l,
                                    chirp.b2)
        kma0_new=np.argmax(np.abs(chirp_new.X(w)))
        rslt.append(chirp_new)
        #print kma0
        #print kma0_new
        assert(kma0_new==0)
    return rslt

def p2_1_3_est(x,ch,kma0,w_name='hann',F_s=16000):
    """
    Estimate the parameters of the chirp using the DDM and output the
    errors.

    x:
        The signal to estimate the paramters of.
    ch:
        The chirp whose parameters we know and we compare to.
    kma0:
        The bin of the centre atom in the estimation scheme.
    w_name:
        The name of the kind of window to use (see: w_dw_sum_cos)
    """
    # Generate window
    w,dw=w_dw_sum_cos(len(x),w_name,norm=False)
    # Estimate parameters
    alpha=ddm_p2_1_3(x,w,dw,kma0)
    # DDM estimation is done with normalized frequency, so correct
    # coefficients
    a0_=np.real(alpha[0][0])
    a1_=np.real(alpha[0][1])*F_s
    a2_=np.real(alpha[0][2])*(F_s**2.)
    b0_=np.imag(alpha[0][0])
    while (b0_ > np.pi):
        b0_ -= 2.*np.pi
    while (b0_ < -np.pi):
        b0_ += 2.*np.pi
    b1_=np.imag(alpha[0][1])*F_s
    while (b1_ > F_s*np.pi):
        b1_ -= F_s*np.pi
    while (b1_ < 0):
        b1_ += F_s*np.pi
    b2_=np.imag(alpha[0][2])*(F_s**2.)
    a0_err=np.abs(ch.a0-a0_)
    a1_err=np.abs(ch.a1-a1_)
    a2_err=np.abs(ch.a2-a2_)
    # Find minimum distance on circle
    b0_err=min([(ch.b0-b0_)%(2.*np.pi),(b0_-ch.b0)%(2.*np.pi)])
    b1_err=min([(ch.b1-b1_)%(np.pi*F_s),(b1_-ch.b1)%(np.pi*F_s)])
    b2_err=np.abs(ch.b2-b2_)
    return (a0_err,a1_err,a2_err,b0_err,b1_err,b2_err)
