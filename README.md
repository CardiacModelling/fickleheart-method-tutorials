# Model calibration with discrepancy 

This repo contains the code for reproducing the results in the examples in the paper "*Considering discrepancy when calibrating a mechanistic electrophysiology model*" by Lei, Ghosh, Whittaker, Aboelkassem, Beattie, Cantwell, Delhaas, Houston, Novaes, Panfilov, Pathmanathan, Riabiz, dos Santos, Worden, Mirams, and Wilkinson.

### Requirements

The code requires Python (3.5+) and the following dependencies:
[PINTS](https://github.com/pints-team/pints#installing-pints),
[Myokit](http://myokit.org/install/),
[Theano](http://deeplearning.net/software/theano/install.html),
[StatsModels](https://www.statsmodels.org/stable/install.html),
[Joblib](https://joblib.readthedocs.io/en/latest/installing.html).

To setup, either run (for Linux/macOS users):
```console
$ bash setup.sh
```
or
navigate to the path where you downloaded this repo and run:
```
$ pip install --upgrade pip
$ pip install .
```

## Ion channel model example

See [ion-channel-models](./ion-channel-models).


## Action potential model example

See [action-potential-models](./action-potential-models).


## Acknowledging this work

If you publish any work based on the contents of this repository please cite:

[PLACEHOLDER]
