#!/usr/bin/env bash

#
# Run `compare-pp-gpcovs.py`.
#

cd ..; source env/bin/activate; cd -

# Compare calibrated model A predictions
python compare-pp-gpcovs.py A sinewave
python compare-pp-gpcovs.py A staircase
python compare-pp-gpcovs.py A ap


# Compare calibrated model B predictions
#python compare-pp-gpcovs.py B sinewave
#python compare-pp-gpcovs.py B staircase
#python compare-pp-gpcovs.py B ap
