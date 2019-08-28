# Ion channel model tutorial

The ion channel that is used throughout this tutorial is the ion channel which carries the current IKr.

### Models

We are interested in selecting the model that 'best describe' the ground truth (model) from our two candidate models.

- Model A (candidate model): Beattie et al. 2018 model.
- Model B (candidate model): Oehmen et al. 2002 model.
- Model C (ground truth model): Di Veroli et al. 2013 model (temperature at 295K; parameters from fitting to Beattie et al. 2018 cell \#5).

### Use of protocol

In this tutorial, we split our protocols into calibration and validation uses.

- Protocol `sinewave`: calibration.
- Protocol `staircase`: validation.
- Protocol `ap`: validation.


## Run the tutorial

1. Run `generate-data.py` to generate synthetic data with iid Gaussian noise (create `data`).
2. Run `fit.py` with arguments `[which_model]` to calibrate the specified model. Alternatively run `fit-all.sh`.
3. Run `predict.py` with arguments `[which_model]` and `[which_predict]` to predict the specified (protocol) data (in `./data`) with the specified model and calibrated model parameters (in `./out`). Alternatively run `predict-all.sh`.
4. Run `compare.py` with arguments `[which_predict]` to compare the predictions of the specified (protocol) data (in `./data`) from the candidate models with the calibrated model parameters (in `./out`). Alternatively run `compare-all.sh`.

### TODO
- Run MCMC.
- Model discrepancy?
- Model selection?

### Output

- `out`: Fitting output etc.
- `fig`: Output generated figures.
- `data`: Data generated from the ground truth model.

### Utilities

- `method`: Contains all the useful methods/functions for this tutorial.
- `mmt-model-files`: Ion channel model in Myokit `mmt` format.
- `protocol-time-series`: Voltage clamp protocols in `csv`, time-series format. Each file has two columns, the first one is time (in [seconds]) and the second column is voltage (in [milliVolts]).

### Tests

- `test-models.py`: Simple test for default model setting and forward model simulations.
