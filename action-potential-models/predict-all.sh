#!/usr/bin/env bash

#
# Run `predict.py` with arguments `[which_model]`, `[which_calibration]`
# and `[which_predict]` to predict the specified (protocol) data (in `./data`)
# with the specified model and calibrated model parameters (in `./out`).
#

# Calibrated tnnp-2004-w predictions
python predict.py tnnp-2004-w stim1hz stim1hz
python predict.py tnnp-2004-w stim1hz stim2hz
python predict.py tnnp-2004-w stim1hz randstim
python predict.py tnnp-2004-w stim1hz hergblock
python predict.py tnnp-2004-w randstim stim1hz
python predict.py tnnp-2004-w randstim stim2hz
python predict.py tnnp-2004-w randstim randstim
python predict.py tnnp-2004-w randstim hergblock

# Calibrated fink-2008 predictions
python predict.py fink-2008 stim1hz stim1hz
python predict.py fink-2008 stim1hz stim2hz
python predict.py fink-2008 stim1hz randstim
python predict.py fink-2008 stim1hz hergblock
python predict.py fink-2008 randstim stim1hz
python predict.py fink-2008 randstim stim2hz
python predict.py fink-2008 randstim randstim
python predict.py fink-2008 randstim hergblock

# Calibrated tnnp-2004 predictions (check identifiability)
python predict.py tnnp-2004 stim1hz stim1hz
python predict.py tnnp-2004 stim1hz stim2hz
python predict.py tnnp-2004 stim1hz randstim
python predict.py tnnp-2004 stim1hz hergblock
python predict.py tnnp-2004 randstim stim1hz
python predict.py tnnp-2004 randstim stim2hz
python predict.py tnnp-2004 randstim randstim
python predict.py tnnp-2004 randstim hergblock

