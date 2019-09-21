# Calibration methods tutorials (Fickle Heart 2019)

This repo contains the code for reproducing the results in the tutorials in the Fickle Heart 2019 calibration method paper. 

### Requirements

The code requires Python (2.7 or 3.6+) and two dependencies:
[PINTS](https://github.com/pints-team/pints#installing-pints) and [Myokit](http://myokit.org/install/).

For Linux/macOS, you may try
```console
$ python3 -m pip install --user --upgrade pip
$ python3 -m pip install --user virtualenv
$ cd /path/to/fickleheart-method-tutorials
$ python3 -m venv env
$ source env/bin/activate
$ pip install --upgrade pip
$ pip3 install myokit                   # Get Myokit, might need external installation for sundials
$ cd /path/to/pints                     # Get PINTS
$ pip install .
$ cd -
$ pip install pymc3                     # Get Theano
$ pip install statsmodels               # Get StatsModels
$ pip install sklearn                   # Get scikit-learn
```


## Ion channel model example

See [ion-channel-models](./ion-channel-models).


## Action potential model example

See [action-potential-models](./action-potential-models).


## Acknowledging this work

If you publish any work based on the contents of this repository please cite:

[PLACEHOLDER]
