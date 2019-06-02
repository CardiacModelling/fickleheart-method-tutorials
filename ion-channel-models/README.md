# Ion channel model tutorial

### Run the tutorial


### Utilities

- `model.py`: PINTS forward model wrapping Myokit solver.
- `mmt-model-files`: Ion channel model in Myokit `mmt` format.
- `protocol-time-series`: Voltage clamp protocols in `csv`, time-series format. Each file has two columns, the first one is time (in [seconds]) and the second column is voltage (in [milliVolts]).

### Tests

- `test-model.py`: Simple test for default model setting and forward model simulations.
