#!/usr/bin/env bash

#
# Run `compare-error-mean.py`, `compare-mean-error.py`, `compare-evidence.py`.
#

cd ..; source env/bin/activate; cd -

# Compare model A
python compare-error-mean.py A
python compare-evidence.py A
python compare-mean-error.py A


# Compare model B
python compare-error-mean.py B
python compare-evidence.py B
python compare-mean-error.py B
