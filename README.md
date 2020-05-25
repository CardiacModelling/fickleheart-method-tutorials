# Model calibration with discrepancy 

This repo contains the code for reproducing the results in the examples in the paper "*Considering discrepancy when calibrating a mechanistic electrophysiology model*" by Lei, Ghosh, Whittaker, Aboelkassem, Beattie, Cantwell, Delhaas, Houston, Novaes, Panfilov, Pathmanathan, Riabiz, dos Santos, Walmsley, Worden, Mirams, and Wilkinson.
[doi:10.1098/rsta.2019.0349](https://doi.org/10.1098/rsta.2019.0349).

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


## Action potential model example

See [action-potential-models](./action-potential-models).


## Ion channel model example

See [ion-channel-models](./ion-channel-models).


## Acknowledging this work

If you publish any work based on the contents of this repository please cite ([CITATION file](CITATION)):

Lei, C.L. _et al._
(2020).
[Considering discrepancy when calibrating a mechanistic electrophysiology model](https://doi.org/10.1098/rsta.2019.0349).
Philosophical Transactions of the Royal Society A, 378: 20190349.
