#!/usr/bin/env python

#
# Run `mcmc.py` with arguments `[which_model]` and `[which_data]` to calibrate
# the specified model with the specified (protocol) data from `./data` using
# MCMC.
#

# Make log dir
mkdir -p log

# Run tnnp-2004-w models
nohup python mcmc.py tnnp-2004-w stim1hz >> log/mcmc-tnnpw-stim1hz.log 2>&1 &
nohup python mcmc.py tnnp-2004-w randstim >> log/mcmc-tnnpw-randstim.log 2>&1 &

# Run fink-2008 models
nohup python mcmc.py fink-2008 stim1hz >> log/mcmc-fink-stim1hz.log 2>&1 &
nohup python mcmc.py fink-2008 randstim >> log/mcmc-fink-randstim.log 2>&1 &

