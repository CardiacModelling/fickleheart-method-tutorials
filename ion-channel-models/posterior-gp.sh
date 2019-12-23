#!/usr/bin/env bash

#
# Run `posterior.py` with arguments `[which_model]` and `[which_predict]` to
# create posterior predictives for the specified (protocol) data (in `./data`)
# with the specified model and MCMC samples of the model parameters (in
# `./out`).
#

cd ..; source env/bin/activate; cd -

# Model A posterior predictive
nohup python posterior-gp.py A sinewave &
sleep 2
nohup python posterior-gp.py A staircase &
sleep 2
nohup python posterior-gp.py A ap &
sleep 2

# Model B posterior predictive
nohup python posterior-gp.py B sinewave &
sleep 2
nohup python posterior-gp.py B staircase &
sleep 2
nohup python posterior-gp.py B ap &
sleep 2
