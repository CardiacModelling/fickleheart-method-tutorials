# Ion channel model tutorial

The ion channel that is used throughout this tutorial is the ion channel which carries the current IKr.

### Models

We are interested in selecting the model that 'best describe' the ground truth (model) from our two candidate models.

- Model A (candidate model): Beattie et al. 2018 model.
- Model B (candidate model): Oehmen et al. 2002 model.
- Model C (ground truth model): Di Veroli et al. 2013 model (temperature at 295K; parameters from fitting to Beattie et al. 2018 cell \#5 data).

### Use of protocol

In this tutorial, we split our protocols into calibration and validation uses.

- Protocol `sinewave`: calibration.
- Protocol `staircase`: validation.
- Protocol `ap`: validation.

### Arguments

- `[which_model]` can be one of `A`, `B`.
- `[which_predict]` can be one of `sinewave`, `staircase`, `AP`.

## Run the tutorial

Before calibration, run `generate-data.py` to generate synthetic data with i.i.d. Gaussian noise (create `data`).

### 1. Calibration with i.i.d. noise assumption
1. Run `fit.py` with arguments `[which_model]` to calibrate the specified model. Alternatively run `fit-all.sh`.
2. Run `predict.py` with arguments `[which_model]` and `[which_predict]` to predict the specified (protocol) data (in `./data`) with the specified model and calibrated model parameters (in `./out`). Alternatively run `predict-all.sh`.
3. Run `compare.py` with arguments `[which_predict]` to compare the predictions of the specified (protocol) data (in `./data`) from the candidate models with the calibrated model parameters (in `./out`). Alternatively run `compare-all.sh`.
4. Run `mcmc.py` with arguments `[which_model]` to run MCMC for the specified model. Alternatively run `mcmc-all.sh`.

### 2. Calibration with discrepancy model: GP(t)
1. Run `fit-gp.py` with arguments `[which_model]` to calibrate the specified model.
2. Run `mcmc-gp.py` with arguments `[which_model]` to run MCMC for the specified model.

### 3. Calibration with discrepancy model: GP(O,V)
1. Run `fit-gp-ov.py` with arguments `[which_model]` to calibrate the specified model.
2. Run `mcmc-gp-ov.py` with arguments `[which_model]` to run MCMC for the specified model.

### 4. Calibration with discrepancy model: ARMA(p,q)
1. Run `mcmc-arma.py` with arguments `[which_model]`, `[arma_p]`, and `[arma_q]` to run MCMC for the specified model, where `[arma_p]` and `[arma_q]` are integers specifying the order of the AR and MA models, respectively.

### Output

- `out`: Fitting output etc.
- `fig`: Output generated figures.
- `data`: Data generated from the ground truth model.

### Utilities

- `method`: Contains all the useful methods/functions for this tutorial.
- `mmt-model-files`: Ion channel models in Myokit `mmt` format.
- `protocol-time-series`: Voltage clamp protocols in `csv`, time-series format. Each file has two columns, the first one is time (in [seconds]) and the second column is voltage (in [milliVolts]).

### Tests

- `test-models.py`: Simple test for default model setting and forward model simulations.

### Others
- `fit-gp-v.py`, `mcmc-gp-v.py`: Run calibration with discrepancy model GP(t,V), which is not shown in the paper.
- `mcmc-arma.py`: Run calibration with discrepancy model ARMA(p,q) model with the invertibility condition, which is not shown in the paper.
