import ddm

# Length of window
N=25
# Relative peak sidelobe height dB
a=-57.
# Mainlobe width in cycles/sample
b=(-.07401*a+1.007)/N
# Design coefficient for -49 > a
W=(-3.81e-2*a+3.16e-1)/N


