#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('./method')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import model as m

"""
Run fit.
"""

model_list = ['A', 'B', 'C']
predict_list = ['sinewave', 'staircase', 'activation', 'ap']

try:
    which_model = sys.argv[1]
    which_predict = sys.argv[2]
except:
    print('Usage: python %s [str:which_model]' % os.path.basename(__file__)
            + ' [str:which_predict]')
    sys.exit()

if which_model not in model_list:
    raise ValueError('Input model %s is not available in the model list' \
            % which_model)

if which_predict not in predict_list:
    raise ValueError('Input data %s is not available in the predict list' \
            % which_predict)

# Get all input variables
import importlib
sys.path.append('./mmt-model-files')
info_id = 'model_%s' % which_model
info = importlib.import_module(info_id)

data_dir = './data'

savedir = './fig/' + info_id
if not os.path.isdir(savedir):
    os.makedirs(savedir)

data_file_name = 'data-%s.csv' % which_predict
print('Predicting ', data_file_name)
saveas = info_id + '-sinewave-' + which_predict

# Protocol
protocol = np.loadtxt('./protocol-time-series/%s.csv' % which_predict,
        skiprows=1, delimiter=',')
protocol_times = protocol[:, 0]
protocol = protocol[:, 1]

# Load data
data = np.loadtxt(data_dir + '/' + data_file_name,
                  delimiter=',', skiprows=1)  # headers
times = data[:, 0]
data = data[:, 1]

# Model
model = m.Model(info.model_file,
        variables=info.parameters,
        current_readout=info.current_list,
        set_ion=info.ions_conc,
        transform=None,
        temperature=273.15 + info.temperature,  # K
        )

# Update protocol
model.set_fixed_form_voltage_protocol(protocol, protocol_times)

# Load calibrated parameters
calloaddir = './out/' + info_id
load_seed = 542811797
fix_idx = [1, 2, 3]
cal_params = []
for i in fix_idx:
    cal_params.append(np.loadtxt('%s/%s-solution-%s-%s.txt' % \
            (calloaddir, 'sinewave', load_seed, i)))

# Predict
predictions = []
for p in cal_params:
    predictions.append(model.simulate(p, times))

# Plot
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 4),
        gridspec_kw={'height_ratios': [1, 3]})
sim_protocol = model.voltage(times)
axes[0].plot(times, sim_protocol, c='#7f7f7f')
axes[0].set_ylabel('Voltage\n(mV)', fontsize=16)
axes[1].plot(times, data, alpha=0.5, label='Data')
for i, p in zip(fix_idx, predictions):
    axes[1].plot(times, p, label='Prediction %s' % i)
axes[1].legend()
axes[1].set_ylabel('Current (pA)', fontsize=16)
axes[1].set_xlabel('Time (ms)', fontsize=16)

plt.subplots_adjust(hspace=0)
plt.savefig('%s/%s' % (savedir, saveas), bbox_inches='tight', dpi=200)
plt.close()

