#!/usr/bin/env bash

### Set up Python virtual env
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install --upgrade pip

### Install dependencies
pip install .                         # Install via setup.py
