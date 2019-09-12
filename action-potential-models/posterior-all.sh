#!/usr/bin/env bash

#
# Run `posterior.py` with arguments `[which_model]`, `[which_calibration]`
# and `[which_predict]` to predict the specified (protocol) data (in `./data`)
# with the specified model and model parameter posteriors (in `./out`).
#

# fink-2008 posterior predictions
python posterior.py fink-2008 stim1hz stim1hz
python posterior.py fink-2008 stim1hz stim2hz
python posterior.py fink-2008 stim1hz randstim
python posterior.py fink-2008 stim1hz hergblock
#python posterior.py fink-2008 randstim stim1hz
#python posterior.py fink-2008 randstim stim2hz
#python posterior.py fink-2008 randstim randstim
#python posterior.py fink-2008 randstim hergblock

# tnnp-2004 posterior predictions (check identifiability)
python posterior.py tnnp-2004 stim1hz stim1hz
python posterior.py tnnp-2004 stim1hz stim2hz
python posterior.py tnnp-2004 stim1hz randstim
python posterior.py tnnp-2004 stim1hz hergblock
#python posterior.py tnnp-2004 randstim stim1hz
#python posterior.py tnnp-2004 randstim stim2hz
#python posterior.py tnnp-2004 randstim randstim
#python posterior.py tnnp-2004 randstim hergblock

