#!/usr/bin/env bash

#
# Run `posterior.py` with arguments `[which_model]` and `[which_predict]` to
# create posterior predictives for the specified (protocol) data (in `./data`)
# with the specified model and MCMC samples of the model parameters (in
# `./out`).
#

cd ..; source env/bin/activate; cd -

# Model A posterior predictive
nohup python posterior-gp-ov.py A sinewave -ov &
sleep 2
nohup python posterior-gp-ov.py A staircase -ov &
sleep 2
nohup python posterior-gp-ov.py A ap -ov &
sleep 2

# Model B posterior predictive
nohup python posterior-gp-ov.py B sinewave -ov &
sleep 2
nohup python posterior-gp-ov.py B staircase -ov &
sleep 2
nohup python posterior-gp-ov.py B ap -ov &
sleep 2
