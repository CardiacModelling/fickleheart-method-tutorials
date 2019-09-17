#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('./method')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pints
import pints.io
import pints.plot

import model as m
import protocol

"""
Run posterior prediction.
"""

model_list = ['tnnp-2004-w', 'fink-2008']
cal_list = ['stim1hz', 'stim2hz', 'randstim']
predict_list = ['stim1hz', 'stim2hz', 'randstim', 'hergblock']

try:
    which_model = sys.argv[1]
    which_cal = sys.argv[2]
    which_predict = sys.argv[3]
except:
    print('Usage: python %s [str:which_model]' % os.path.basename(__file__)
            + ' [str:which_calibration] [str:which_predict]')
    sys.exit()

if which_model not in model_list:
    raise ValueError('Input model %s is not available in the model list' \
            % which_model)

if which_cal not in cal_list:
    raise ValueError('Input calibration %s is not available in the data' \
            'list' % which_cal)

if which_predict not in predict_list:
    raise ValueError('Input data %s is not available in the predict list' \
            % which_predict)

loaddir = './out/mcmc-' + which_model
loadas = which_model + '-' + which_cal

savedir = './fig/mcmc-' + which_model
if not os.path.isdir(savedir):
    os.makedirs(savedir)
saveas = which_model + '-' + which_cal + '-' + which_predict

parameter_names = m.parameter_names + [r'$\sigma$']

# Load fitting
load_seed = 542811797
f = '%s/%s/%s-solution-%s-%s.txt' \
        % ('./out', which_model, which_cal, load_seed, 1)
p = np.loadtxt(f)
x0 = np.append(p, np.NaN)

# Load samples
chains = pints.io.load_samples('%s/%s-chain.csv' % (loaddir, loadas), 3)
chains = np.asarray(chains)

n_iter = len(chains[0])
chains = chains[:, int(0.5 * n_iter)::5, :]

# Just double checking, should all be the same
_, axes = pints.plot.pairwise(chains[0], kde=False, ref_parameters=x0)
for i in range(len(axes)):
    axes[i, 0].set_ylabel(parameter_names[i], fontsize=32)
    axes[-1, i].set_xlabel(parameter_names[i], fontsize=32)
plt.tight_layout()
plt.savefig('%s/%s-pairwise.png' % (savedir, saveas), bbox_inches='tight')
plt.close('all')

# Posterior predictions
# Load data
data_dir = './data'
data_file_name = 'data-%s.csv' % which_predict
data = np.loadtxt(data_dir + '/' + data_file_name,
                  delimiter=',', skiprows=1)  # headers
times = data[:, 0]
data = data[:, 1:]
if which_predict == 'hergblock':
    data = data.T
else:
    data = [data]

# Protocol
stim_list = {
        'stim1hz': protocol.stim1hz,
        'stim2hz': protocol.stim2hz,
        'randstim': protocol.randstim,
        'hergblock': protocol.stim1hz_hergblock,
        }
stim_seq = stim_list[which_predict]

# Model
model = m.Model(
        './mmt-model-files/%s.mmt' % which_model,
        stim_seq=stim_seq,
        )
model.set_name(which_model)

# Predict
thinning = max(1, int(len(chains[0]) / 200))
if which_predict == 'hergblock':
    prediction = []
    for params in chains[0][::thinning, :model.n_parameters()]:
        prediction.append(protocol.hergblock_simulate(model, params, times))
    mean_values = np.mean(prediction, axis=0)
else:
    prediction = []
    for params in chains[0][::thinning, :model.n_parameters()]:
        prediction.append(model.simulate(params, times))
    mean_values = np.mean(prediction, axis=0)

# Plot
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 4))
for i, d in enumerate(data):
    axes.plot(times, d, c='C' + str(i), alpha=0.5, label='data')
if which_predict == 'hergblock':
    for predicted_values in prediction:
        for i, p in enumerate(predicted_values):
            axes.plot(times, p, c='C' + str(i), alpha=0.2, ls='--')
else:
    for predicted_values in prediction:
        axes.plot(times, predicted_values, c='C0', alpha=0.2, ls='--')
# axes.legend()
axes.set_ylabel('Voltage (mV)')
axes.set_xlabel('Time (ms)')
plt.subplots_adjust(hspace=0)
plt.savefig('%s/%s-pp.png' % (savedir, saveas), bbox_inches='tight')
plt.close()
