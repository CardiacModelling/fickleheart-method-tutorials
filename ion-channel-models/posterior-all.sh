#!/usr/bin/env bash

#
# Run `posterior.py` with arguments `[which_model]` and `[which_predict]` to
# create posterior predictives for the specified (protocol) data (in `./data`)
# with the specified model and MCMC samples of the model parameters (in
# `./out`).
#

# Model A posterior predictive
python posterior.py A sinewave
python posterior.py A staircase
python posterior.py A ap

python posterior-arma.py A sinewave
python posterior-arma.py A staircase
python posterior-arma.py A ap

python posterior-gp.py A sinewave
python posterior-gp.py A staircase
python posterior-gp.py A ap

python posterior-gp-tv.py A sinewave
python posterior-gp-tv.py A staircase
python posterior-gp-tv.py A ap

python posterior-gp-ov.py A sinewave -v
python posterior-gp-ov.py A staircase -v
python posterior-gp-ov.py A ap -v

python posterior-gp-ov.py A sinewave -ov
python posterior-gp-ov.py A staircase -ov
python posterior-gp-ov.py A ap -ov

# Model B posterior predictive
python posterior.py B sinewave
python posterior.py B staircase
python posterior.py B ap

python posterior-arma.py B 2 2 sinewave
python posterior-arma.py B 2 2 staircase
python posterior-arma.py B 2 2 ap

python posterior-gp.py B sinewave
python posterior-gp.py B staircase
python posterior-gp.py B ap

python posterior-gp-tv.py B sinewave
python posterior-gp-tv.py B staircase
python posterior-gp-tv.py B ap

python posterior-gp-ov.py B sinewave -v
python posterior-gp-ov.py B staircase -v
python posterior-gp-ov.py B ap -v

python posterior-gp-ov.py B sinewave -ov
python posterior-gp-ov.py B staircase -ov
python posterior-gp-ov.py B ap -ov

