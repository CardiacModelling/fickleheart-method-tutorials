# Ion channel model tutorial

The ion channel that is used throughout this tutorial is the ion channel which carries the current IKr.

### Models

We are interested in selecting the model that 'best describe' the ground truth (model) from our two candidate models.

- Model A (candidate model): Beattie et al. 2018 model (parameters from cell \#5).
- Model B (candidate model): Oehmen et al. 2002 model.
- Model C (ground truth model): Mazhari et al. 2001 model.

### Run the tutorial


### Output

- `out`: Fitting output etc.
- `fig`: Output generated figures.

### Utilities

- `method`: Contains all the useful methods/functions for this tutorial.
- `mmt-model-files`: Ion channel model in Myokit `mmt` format.
- `protocol-time-series`: Voltage clamp protocols in `csv`, time-series format. Each file has two columns, the first one is time (in [seconds]) and the second column is voltage (in [milliVolts]).

### Tests

- `test-models.py`: Simple test for default model setting and forward model simulations.

