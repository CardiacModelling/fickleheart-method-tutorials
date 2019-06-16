# Action potential model tutorial

A typical fitting of cardiac action potential model is by scaling/updating the maximum conductance parameters of an existing action potential model.
This method usually assumes that the underlying kinetics of each ion channel current within the candidate model is correct and perfect (i.e. match perfectly to the ground truth).
Here we explicitly impose that the underlying kinetics of each ion channel current is _imperfect_.
Then we will _ignore_ the discrepancy and proceed with our analysis with the incorrect assumption (i.e. without acknowledging model discrepancy).
Finally, we compare the prediction of those calibrated candidate models under our context of use (COU) to the ground truth (model).

### Models

- Model Fink 2008 (candidate model): Fink et al. 2008 model.
- Model TNNP 2004 wrong (candidate model): modified ten Tusscher et al. 2004 model.
- Model TNNP 2004 (ground truth model): ten Tusscher et al. 2004 model.

### Use of protocol

In this tutorial, we split our protocols into calibration and COU.
Note that here we do not have validation protocol, as to emphasise the importance of validation.

- Protocol `stim1hz`: calibration.
- Protocol `stim2hz`: calibration.
- Protocol `randstim`: calibration.
- Protocol `hergblock`: COU prediction.
- Protocol `current`: COU prediction.

## Run the tutorial

1. Run `generate-data.py` to generate synthetic data with iid Gaussian noise (create `data`).
2. Run `fit.py` with arguments `[which_model]` and `[which_data]` to calibrate the specified model with the specified (protocol) data from `./data`.
3. Run `predict.py` with arguments `[which_model]`, `[which_calibration]` and `[which_predict]` to predict the specified (protocol) data (in `./data`) with the specified model and calibrated model parameters (in `./out`).

## TODO

- For 2., can also try history matching type of method? Though doubt will make a difference.
- Run MCMC, to show we are 'confident' wrongly.
- Do `current` prediction.

### Output

- `out`: Fitting output etc.
- `fig`: Output generated figures.
- `data`: Data generated from the ground truth model.

### Utilities

- `method`: Contains all the useful methods/functions for this tutorial.
- `mmt-model-files`: Ion channel model in Myokit `mmt` format.

### Tests

- `test-models.py`: Simple test for default model setting and forward model simulations.

