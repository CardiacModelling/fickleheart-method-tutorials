#!/usr/bin/env bash

#
# Run `predict.py` with arguments `[which_model]` and `[which_predict]` to
# predict the specified (protocol) data (in `./data`) with the specified model
# and calibrated model parameters (in `./out`).
#

# Calibrated model A predictions
python predict.py A sinewave
python predict.py A staircase
python predict.py A ap

# Calibrated model B predictions
python predict.py B sinewave
python predict.py B staircase
python predict.py B ap

