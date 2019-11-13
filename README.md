# Calibration methods tutorials (Fickle Heart 2019)

This repo contains the code for reproducing the results in the tutorials in the Fickle Heart 2019 calibration method paper. 

### Requirements

The code requires Python (3.5+) and the following dependencies:
- [PINTS](https://github.com/pints-team/pints#installing-pints)
- [Myokit](http://myokit.org/install/)
- [Theano](http://deeplearning.net/software/theano/install.html)
- [StatsModels](https://www.statsmodels.org/stable/install.html)
- [Joblib](https://joblib.readthedocs.io/en/latest/installing.html)

For Linux/macOS, you may try
```console
$ cd /path/to/fickleheart-method-tutorials
$ ### Set up Python virtual env
$ python3 -m pip install --user --upgrade pip
$ python3 -m pip install --user virtualenv
$ python3 -m venv env
$ source env/bin/activate
$ pip install --upgrade pip
$ ### Install dependencies
$ pip3 install myokit                   # Get Myokit, might need external installation for sundials
$ pip3 install git+https://github.com/pints-team/pints  # Get PINTS
$ pip3 install Theano                   # Get Theano
$ pip3 install statsmodels              # Get StatsModels
$ pip3 install joblib                   # Get Joblib
```


## Ion channel model example

See [ion-channel-models](./ion-channel-models).


## Action potential model example

See [action-potential-models](./action-potential-models).


## Acknowledging this work

If you publish any work based on the contents of this repository please cite:

[PLACEHOLDER]
