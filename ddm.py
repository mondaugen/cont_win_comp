# The distribution derivative method and windows it requires

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import cvxopt

FONT_OPT=dict()
FONT_OPT['dafx']={
        'family':'serif',
        'sans-serif':'Times'
}

def eig_pow_method(A,d=1.e-6,maxiters=1000):
    """
    Compute the greatest eigenvalue of A and its corresponding eigenvector using
    the power method. Stops when maxiters reached or difference between last
    greatest absolute value and current greatest absolute value less than d.
    """
    N=A.shape[1]
    b=np.zeros(N,dtype='complex')
    b[0]=1.
    b/=LA.norm(b)
    lst_max=np.max(np.abs(b))
    for i in xrange(maxiters):
        b=np.inner(A,b)
        b/=LA.norm(b)
        if np.abs(lst_max - np.max(np.abs(b))) < d:
            break
        lst_max=np.max(np.abs(b))
    return (np.vdot(b,np.inner(A,b)),b)

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
        elif (a=='naive-hamming'):
            # Called naive because this will not include the derivative of the
            # rectangular window, which is non-zero at the window edges
            c=np.r_[0.54,0.46]
        elif (a=='naive-min-blackman-4'):
            c=np.r_[0.35875,0.48829,0.14128,0.01168]
        elif (a=='prolate-0.008-approx-5'):
            c=np.r_[3.128258203861262743e-01,
                    4.655346501582137142e-01,
                    1.851027605614040672e-01,
                    3.446534984178639682e-02,
                    2.071419052469766937e-03]
        elif (a=='prolate-0.77-approx-47'):
            # Does the number of figures contribute significantly to the
            # accuracy?
            c=np.r_[6.815625423533416827e-02,
                    1.343413407697203465e-01,
                    1.285968082160944270e-01,
                    1.195619690697207610e-01,
                    1.079670679318169152e-01,
                    9.469240652932217617e-02,
                    8.065884608008552781e-02,
                    6.672467071524876281e-02,
                    5.360436978540508018e-02,
                    4.181885562354893671e-02,
                    3.167949690383391192e-02,
                    2.330191778479831152e-02,
                    1.664111594502999322e-02,
                    1.153766107795279119e-02,
                    7.765363769402260606e-03,
                    5.073125727673936242e-03,
                    3.216749477092172276e-03,
                    1.979436174490603606e-03,
                    1.181952653281245508e-03,
                    6.847647303719338633e-04,
                    3.848668599270928963e-04,
                    2.098211237310444353e-04,
                    1.109419375406991352e-04,
                    5.688331526814708938e-05,
                    2.827794498282796775e-05,
                    1.362737098580978749e-05,
                    6.365042844173701925e-06,
                    2.880948326056731291e-06,
                    1.263369519424718326e-06,
                    5.366584133144254092e-07,
                    2.207728978301219456e-07,
                    8.793812084883303941e-08,
                    3.390713653010824865e-08,
                    1.265266172182148338e-08,
                    4.568137255870320335e-09,
                    1.595319231469439323e-09,
                    5.387492617615924091e-10,
                    1.758854467412710503e-10,
                    5.549388693644718546e-11,
                    1.691607734986330884e-11,
                    4.980293127726517115e-12,
                    1.415506327451228367e-12,
                    3.880994179517495547e-13,
                    1.024294291393307026e-13,
                    2.578916000719680285e-14,
                    5.959884173538302433e-15,
                    1.137696503621815197e-15]
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

def ddm_p2_1_R(x,w,dw,kma0,R=3):
    """
    Compute parameters of 2nd order polynomial using R bins surrounding the
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
        corresponding to this index and the (N-1) surrounding ones.
    R:
        the number of atoms to use. Must be odd.

    Returns

    a:
        a vector containing the estimated parameters.
    """
    if (R % 2 != 1):
        raise Error('R must be odd.')

    N_w=len(w)
    nx0=np.arange(N_w)
    x0=x[nx0]
    #k_plus=(np.arange((R-1)/2) + kma0 + 1) % N_w
    #k_minus=((kma0 - np.arange((R-1)/2) - 1) % N_w)[::-1]
    k_plus=(np.arange((R-1)/2) + kma0 + 1)
    k_minus=((kma0 - np.arange((R-1)/2) - 1))[::-1]
    k_atoms=np.concatenate((k_minus,[kma0],k_plus))
    # Shitty fourier basis, why?
    F=np.exp(-2.*np.pi*1j*np.outer(k_atoms,nx0)/float(N_w))
    Xp1w=np.dot(F,x0*w)
    Xp2w=np.dot(F,2.*nx0*x0*w)
    Xdw_=np.dot(F,x0*dw)
    #Xp1w=np.fft.fft(x0*w)
    #Xp2w=np.fft.fft(2.*nx0*x0*w)
    #Xdw_=np.fft.fft(x0*dw)
    Xdw=Xp1w*(-2.*np.pi*1j*k_atoms/float(N_w))+Xdw_
    A=np.c_[
            np.r_[
#                Xp1w[k_atoms]
                Xp1w
                ],
            np.r_[
#                Xp2w[k_atoms]
                Xp2w
                ],
            ]
    c=np.c_[
            np.r_[
#                Xdw[k_atoms]
                Xdw
                ]
            ]
    result=[]
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

def p2_1_3_est(x,ch,kma0,w_name='hann',F_s=16000,R=3):
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

    Returns:

        A tuple containing the absolute error for each parameter.
        A tuple containing the estimated parameters.
    """
    # Generate window
    w,dw=w_dw_sum_cos(len(x),w_name,norm=False)
    # Estimate parameters
    alpha=ddm_p2_1_R(x,w,dw,kma0,R)
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
    return ((a0_err,a1_err,a2_err,b0_err,b1_err,b2_err),
                (a0_,a1_,a2_,b0_,b1_,b2_))

def psw_design(N,W):
    """
    Design a prolate spheroidal window of length N using parameter W.
    """
    i=np.arange(N,dtype='double')
    A=np.diag(np.cos(2*np.pi*W)*(((N-1.)/2.-i)**2.))
    A+=np.diag(0.5*i[1:]*(N-i[1:]),-1)
    A+=np.diag(0.5*i[1:]*(N-i[1:]),1)
    #(l,v)=ddm.eig_pow_method(A)
    (L,V)=np.linalg.eig(A)
    v=V[:,np.argmax(np.abs(L))]
    return v

def bw_bins(w,A_bw=10.**(-3./20.),k_hint=0,k_max=0):
    """
    Find the bin closest to k_hint that has ampltiude A_bw.
    This is useful for finding the 3db bandwidth of windows.
    Note that this amplitude is normalized by the DTFT at the maximum bin
    (k_max).

    Arguments:
        
        w:
            the window whose DTFT will be used to find the bin.
        A_bw:
            the amplitude of the desired bin.
        k_hint:
            the bin from which the search will start, can be float.
        k_max:
            the bin where the DTFT will be maximum, can be float.

    Returns:
        k, the bin whose amplitude is closest to A_bw.
    """
    # Fourier transform at maxmimum bin
    N=len(w)
    n=np.arange(N)
    W_0=np.inner(w,np.exp(-2.*np.pi*1j*n*float(k_max)/N))
    def _f(x):
        result = np.inner(w,np.exp(-2.*np.pi*1j*n*x/float(N)))/W_0 
        result = result * np.conj(result)
        return  np.real(result - A_bw**2.)
    def _df(x):
        W_k=np.inner(w,np.exp(-2.*np.pi*1j*n*x/float(N)))
        result1 = (np.inner(w,-2.*np.pi*1j*n/float(N)*np.exp(-2.*np.pi*1j*n*x/float(N)))/W_0) 
        result1 *= np.conj(W_k/W_0)
        result2 = (np.inner(w,2.*np.pi*1j*n/float(N)*np.exp(2.*np.pi*1j*n*x/float(N)))/W_0) 
        result2 *= W_k/W_0
        return np.real(result1+result2)
    k_opt = spopt.newton(_f,k_hint,_df)
#    k_opt = spopt.newton(_f,k_hint)
    return k_opt

def cos_approx_win(v,M,c={'c1'}):
    """
    Find the M term harmonically related cosine approximation of v.
    c is a set of constraints.

    v:
        A vector representing the window to approximate.
        Usually v will have even length so that its periodization is
        symmetric around v[0] (it is an even function).
    M:
        The number of terms in the approximation.
    c:
        A set of keywords indicating constraints.
        By default {'c1'}, which makes the window once differentiable.
        (This is currently the only possible constraint).

    Returns:

    a:
        The M coefficients of the window.

    """
    N=len(v)
    n=np.arange(N)
    m=np.arange(M)
    # Cosines to sum to get window
    C=np.cos(2*np.pi*np.outer(n,m)/N)*np.power(-1.,np.arange(M))
    d=dict()
    d['P']=cvxopt.matrix(2.*np.dot(C.T,C))
    d['q']=cvxopt.matrix(-2.*np.dot(C.T,v.reshape(N,1)))
    if 'c1' in c:
        A=np.concatenate((np.ones((1,M)),C[0,:].reshape((1,M))),axis=0)
        b=np.array([[1.],[0.]])
        d['A']=cvxopt.matrix(A)
        d['b']=cvxopt.matrix(b)
    a=cvxopt.solvers.qp(**d)['x']
    a=np.array(a).flatten()
    return a

def crlb_pq(a,t,sig2):
    """
    Returns the CR lower bounds for the parameters a0, ..., a(q-1).

    a:
        The coefficients of the log-amplitude polynomial, i.e.,
        exp(a0 + a1 * t + a2 * t^2 + ... + a(q-1) * t^(q-1)).
        These values should be real.
    t:
        The values of t over which to sum.
    sig2:
        The variance of the noise.

    Returns a vector v of length q, v[0] is CRLB of paramater a0
    """
    # Build Fischer matrix
    q=len(a)
    F=np.zeros((q,q),dtype='float')
    for i in xrange(q):
        for j in xrange(q):
            F[i,j]=2./sig2 * np.sum(np.power(t,i+j) *
                    np.exp(2*np.polyval(a[::-1],t)))
    # Return diagonal of inverse
    return np.diag(np.linalg.inv(F))

def localmax(x):
    """
    Returns local maxima of x and the mask used to get them. Includes endpoints
    even though not necessarily local maxima.
    """
    x=np.array(x).flatten()
    xmask=np.r_[True,(x[1:]>x[:-1])[:-1]&(x[:-1]>x[1:])[1:],True]
    return (x[xmask],xmask)
