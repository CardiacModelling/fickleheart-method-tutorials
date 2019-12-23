#!/usr/bin/env bash

#
# Run `compare-pp.py`.
#

cd ..; source env/bin/activate; cd -

# Compare calibrated model A predictions
python compare-pp.py A sinewave
python compare-pp.py A staircase
python compare-pp.py A ap


# Compare calibrated model B predictions
python compare-pp.py B sinewave
python compare-pp.py B staircase
python compare-pp.py B ap
