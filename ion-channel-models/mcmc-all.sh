#!/usr/bin/env python

#
# Run `mcmc.py` with arguments `[which_model]` to run MCMC for the specified
# model.
#

# Make log dir
mkdir -p log

# Fit model A
nohup python mcmc.py A 10 >> log/mcmc-A.log 2>&1 &

# Fit model B
nohup python mcmc.py B 10 >> log/mcmc-B.log 2>&1 &

