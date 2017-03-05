import numpy as np
import numpy.linalg
import ddm
import matplotlib.pyplot as plt
import sigmod as sm

N=512

w=ddm.w_dw_sum_cos(512,a='hanning')[0]

k_opt=ddm.bw_bins(w,k_hint=1.)

print '3db bandwidth: %g' % (k_opt * 2.,)
