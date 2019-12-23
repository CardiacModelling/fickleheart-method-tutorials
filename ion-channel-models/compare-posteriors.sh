#!/usr/bin/env bash

#
# Run `compare-posteriors.py`.
#

cd ..; source env/bin/activate; cd -

# Compare model A
python compare-posteriors.py A

# Compare model B
python compare-posteriors.py B
