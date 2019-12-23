#!/usr/bin/env bash

#
# Run `compare.py` with arguments `[which_predict]` to compare the predictions
# of the specified (protocol) data (in `./data`) from the two models with
# calibrated model parameters (in `./out`).
#

# Compare calibrated model A, B predictions
python compare.py sinewave
python compare.py staircase
python compare.py ap

