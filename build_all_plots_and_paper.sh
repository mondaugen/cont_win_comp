#!/bin/bash

# clone signal_modeling repo
if [ ! -d ../signal_modeling ]
then
    (cd ..; git clone https://github.com/mondaugen/signal_modeling)
fi

# set python path
source ./set_python_path.sh

# build plot finding optimal continuous approximation to prolate window
python ./search_dpw_bw_m.py

# build plot finding comparing the performance of different windows in noise 
python ./ddm_snr_win_comp.py

# build plot comparing the estimation of two superimposed components using
# different windows
python ./comp_offset_chirp_est_err.py
