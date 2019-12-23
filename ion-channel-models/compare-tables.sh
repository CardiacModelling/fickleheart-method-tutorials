#!/usr/bin/env bash

#
# Run `compare-error-mean.py`, `compare-mean-error.py`, `compare-evidence.py`.
#
-rw-rw-r--  1 chon chon  4406 Dec 21 23:36 compare-error-mean.py
-rw-rw-r--  1 chon chon  3548 Dec 21 23:36 compare-evidence.py
-rw-rw-r--  1 chon chon  3721 Dec 21 23:36 compare-mean-error.py

cd ..; source env/bin/activate; cd -

# Compare model A
python compare-error-mean.py A
python compare-evidence.py A
python compare-mean-error.py A


# Compare model B
python compare-error-mean.py B
python compare-evidence.py B
python compare-mean-error.py B
