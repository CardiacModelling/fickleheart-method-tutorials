#!/usr/bin/env python

#
# Run `fit.py` with arguments `[which_model]` and `[which_data]` to calibrate
# the specified model with the specified (protocol) data from `./data`.
#

# Make log dir
mkdir -p log

# Fit tnnp-2004-w models
nohup python fit.py tnnp-2004-w stim1hz 10 >> log/fit-tnnpw-stim1hz.log 2>&1 &
nohup python fit.py tnnp-2004-w randstim 10 >> log/fit-tnnpw-randstim.log 2>&1 &

# Fit fink-2008 models
nohup python fit.py fink-2008 stim1hz 10 >> log/fit-fink-stim1hz.log 2>&1 &
nohup python fit.py fink-2008 randstim 10 >> log/fit-fink-randstim.log 2>&1 &

# Run tnnp-2004 models (check identifiability)
nohup python fit.py tnnp-2004 stim1hz 10 >> log/fit-tnnp-stim1hz.log 2>&1 &
nohup python fit.py tnnp-2004 randstim 10 >> log/fit-tnnp-randstim.log 2>&1 &
