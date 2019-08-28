#!/usr/bin/env bash

#
# Run `posterior.py` with arguments `[which_model]`, `[which_calibration]`
# and `[which_predict]` to predict the specified (protocol) data (in `./data`)
# with the specified model and model parameter posteriors (in `./out`).
#

# tnnp-2004-w posterior predictions
#python posterior.py tnnp-2004-w stim1hz stim1hz
#python posterior.py tnnp-2004-w stim1hz stim2hz
#python posterior.py tnnp-2004-w stim1hz randstim
#python posterior.py tnnp-2004-w stim1hz hergblock
#python posterior.py tnnp-2004-w randstim stim1hz
#python posterior.py tnnp-2004-w randstim stim2hz
#python posterior.py tnnp-2004-w randstim randstim
#python posterior.py tnnp-2004-w randstim hergblock

# fink-2008 posterior predictions
python posterior.py fink-2008 stim1hz stim1hz
python posterior.py fink-2008 stim1hz stim2hz
python posterior.py fink-2008 stim1hz randstim
python posterior.py fink-2008 stim1hz hergblock
#python posterior.py fink-2008 randstim stim1hz
#python posterior.py fink-2008 randstim stim2hz
#python posterior.py fink-2008 randstim randstim
#python posterior.py fink-2008 randstim hergblock

