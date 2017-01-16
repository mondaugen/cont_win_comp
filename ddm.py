# The distribution derivative method and windows it requires

import numpy as np

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
