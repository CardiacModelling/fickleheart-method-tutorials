# Action potential model tutorial

A typical fitting of cardiac action potential model is by scaling/updating the maximum conductance parameters of an existing action potential model.
This method usually assumes that the underlying kinetics of each ion channel current is correct and perfect.
Here we explicitly impose that the underlying kinetics of each ion channel current is _imperfect_.
Then we will _ignore_ the discrepancy and process will our analysis with the incorrect assumption.

### Models

- Model Fink 2008 (candidate model): Fink et al. 2008 model.
- Model TNNP 2004 w (candidate model): modified ten Tusscher et al. 2004 model.
- Model TNNP 2004 (ground truth model): ten Tusscher et al. 2004 model.

### Use of protocol

In this tutorial, we split our protocols into calibration and COU uses.
Note that here we do not have validation protocol, as to emphasise the importance of validation.

- Protocol `stim1hz`: calibration.
- Protocol `stim2hz`: calibration.
- Protocol `randstim`: calibration.
- Protocol `hergblock`: COU prediction.
- Protocol `current`: COU prediction.

## Run the tutorial

1. Run `generate-data.py` to generate synthetic data with iid Gaussian noise (create `data`).

### Output

- `out`: Fitting output etc.
- `fig`: Output generated figures.

### Utilities

- `method`: Contains all the useful methods/functions for this tutorial.
- `mmt-model-files`: Ion channel model in Myokit `mmt` format.

### Tests

- `test-models.py`: Simple test for default model setting and forward model simulations.

