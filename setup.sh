#!/usr/bin/env bash

### Set up Python virtual env
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install --upgrade pip

### Install dependencies
pip3 install myokit                   # Get Myokit, might need external installation for sundials
pip3 install pints                    # Get PINTS
pip3 install Theano                   # Get Theano
pip3 install statsmodels              # Get StatsModels
pip3 install joblib                   # Get Joblib
