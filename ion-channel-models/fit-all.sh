#!/usr/bin/env python

#
# Run `fit.py` with arguments `[which_model]` to calibrate the specified model.
#

# Make log dir
mkdir -p log

# Fit model A
nohup python fit.py A 10 > log/fit-A.log 2>&1 &

sleep 5

# Fit model B
nohup python fit.py B 10 > log/fit-B.log 2>&1 &

