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

from scipy.stats import norm as scipy_stats_norm
from sparse_gp_custom_likelihood import _create_theano_conditional_graph_voltage

import model as m

"""
Posterior predictive with (sparse) GP discrepancy model.
"""

model_list = ['A', 'B', 'C']
predict_list = ['sinewave', 'staircase', 'activation', 'ap']

np.random.seed(101)  # fix seed for prediction

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

savedir = './fig/mcmc-' + info_id + '-gp-tv'
if not os.path.isdir(savedir):
    os.makedirs(savedir)
if not os.path.isdir(savedir + '/raw'):
    os.makedirs(savedir + '/raw')

data_file_name = 'data-%s.csv' % which_predict
print('Predicting ', data_file_name)
saveas = info_id + '-sinewave-' + which_predict

loaddir = './out/mcmc-' + info_id + '-gp-tv'
loadas = info_id + '-sinewave'

# Protocol
protocol_train = np.loadtxt('./protocol-time-series/sinewave.csv', skiprows=1,
        delimiter=',')
protocol_train_times = protocol_train[:, 0]
protocol_train = protocol_train[:, 1]

protocol = np.loadtxt('./protocol-time-series/%s.csv' % which_predict,
        skiprows=1, delimiter=',')
protocol_times = protocol[:, 0]
protocol = protocol[:, 1]

# Load data
data_train = np.loadtxt(data_dir + '/data-sinewave.csv',
                  delimiter=',', skiprows=1)  # headers
times_train = data_train[:, 0]
data_train = data_train[:, 1]

data = np.loadtxt(data_dir + '/' + data_file_name,
                  delimiter=',', skiprows=1)  # headers
times = data[:, 0]
data = data[:, 1]

# Inducing or speudo points for the FITC GP
inducing_times_train = times_train[::1000]

# Load model
model_train = m.Model(info.model_file,
        variables=info.parameters,
        current_readout=info.current_list,
        set_ion=info.ions_conc,
        transform=None,
        temperature=273.15 + info.temperature,  # K
        )
# Update protocol to training protocol
model_train.set_fixed_form_voltage_protocol(protocol_train,
        protocol_train_times)

model = m.Model(info.model_file,
        variables=info.parameters,
        current_readout=info.current_list,
        set_ion=info.ions_conc,
        transform=None,
        temperature=273.15 + info.temperature,  # K
        )
# Update protocol to predicting protocol
model.set_fixed_form_voltage_protocol(protocol, protocol_times)

# Simulate voltage
voltage = model.voltage(times)

# Load MCMC results
ppc_samples = pints.io.load_samples('%s/%s-chain_0.csv' % (loaddir, loadas))


# Bayesian prediction of GP discrepancy based on the variance identity
# -----------------------------------------------------------------------------
# Posterior predictive in light of the discontinuity GP. Like the ARMAX case
# we use the variance identity here. Currently this is setup to use the same
# protocol for training and validation. Please change this accordingly.
# How this works: Basically we want
# \int GP(Current_validation|Current_training, Data, gp_params, ode_params) 
#     d gp_params d ode_params .
# We get GP(Current_validation|Current_training, Data, gp_params, ode_params)
# as Normal distribution, see ppc_mean, ppc_var, for a single sample of
# (gp_params, ode_params). To propagate the uncertainty fully we then use the
# same Variance identity for ARMAX to integrate out (gp_params, ode_params).
# -----------------------------------------------------------------------------
ppc_size = np.size(ppc_samples, axis=0)
gp_ppc_mean = []
gp_ppc_var = []
model_ppc_mean = []
gp_only_ppc_mean = []

training_data = data_train.reshape((-1,))
t_training_protocol = times_train.reshape((-1, 1)) 
ind_t = inducing_times_train.reshape((-1, 1))
t_valid_protocol = times.reshape((-1, 1)) 
ppc_sampler_mean, ppc_sampler_var = _create_theano_conditional_graph_voltage(
        training_data, t_training_protocol, ind_t, t_valid_protocol)
nds = 4  # Number of discrepancy parameters

#for ind in random.sample(range(0, np.size(ppc_samples, axis=0)), 100):
for ind in np.random.choice(range(0, ppc_size), 100, replace=False):
    # Expecting these parameters can be used for simulation
    ode_params = ppc_samples[ind, :-nds]
    # Expecting these GP parameters are untransformed
    _sigma, _rho_t, _rho_v, _ker_sigma = ppc_samples[ind,-nds:]
    _rho = np.append(_rho_t, _rho_v)

    # Simulate
    current_training_protocol = model_train.simulate(ode_params, times_train)
    current_valid_protocol = model.simulate(ode_params, times)

    # Compute mean and var
    ppc_mean = ppc_sampler_mean(current_training_protocol,
            current_valid_protocol, _rho, _ker_sigma, _sigma)
    ppc_var = ppc_sampler_var(current_training_protocol,
            current_valid_protocol, _rho, _ker_sigma, _sigma)

    gp_ppc_mean.append(ppc_mean)
    gp_ppc_var.append(ppc_var)
    model_ppc_mean.append(current_valid_protocol)
    gp_only_ppc_mean.append(ppc_mean - current_valid_protocol)

n_sd = scipy_stats_norm.ppf(1. - .05 / 2.)

# Model + GP
ppc_mean = np.mean(gp_ppc_mean, axis=0)
var1 = np.mean(gp_ppc_var, axis=0)
var2_1 = np.mean(np.power(gp_ppc_mean, 2), axis=0)
var2_2 = np.power(np.mean(gp_ppc_mean, axis=0), 2)
ppc_sd = np.sqrt(var1 + var2_1 - var2_2)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
        gridspec_kw={'height_ratios': [1, 2]})
axes[0].plot(times, voltage, c='#7f7f7f')
axes[0].set_ylabel('Voltage (mV)')
axes[1].plot(times, data, alpha=0.5, label='data')
axes[1].plot(times, ppc_mean, label='Mean')
axes[1].plot(times, ppc_mean + n_sd * ppc_sd, '-', color='blue', lw=0.5,
        label='95% C.I.')
axes[1].plot(times, ppc_mean - n_sd * ppc_sd, '-', color='blue', lw=0.5)
axes[1].legend()
axes[1].set_ylabel('Current (pA)')
axes[1].set_xlabel('Time (ms)')
plt.subplots_adjust(hspace=0)
plt.savefig('%s/%s-pp.png' % (savedir, saveas), dpi=200,
        bbox_inches='tight')
plt.close()

# Model only
model_mean = np.mean(model_ppc_mean, axis=0)
var1_1 = np.mean(np.power(model_ppc_mean, 2), axis=0)
var1_2 = np.power(np.mean(model_ppc_mean, axis=0), 2)
model_sd = np.sqrt(var1_1 - var1_2)
print(np.sum(np.abs(model_sd - np.std(model_ppc_mean, axis=0))))

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
        gridspec_kw={'height_ratios': [1, 2]})
axes[0].plot(times, voltage, c='#7f7f7f')
axes[0].set_ylabel('Voltage (mV)')
axes[1].plot(times, data, alpha=0.5, label='data')
axes[1].plot(times, model_mean, label='Mean')
axes[1].plot(times, model_mean + n_sd * model_sd, '-', color='blue', lw=0.5,
        label='95% C.I.')
axes[1].plot(times, model_mean - n_sd * model_sd, '-', color='blue', lw=0.5)
axes[1].legend()
axes[1].set_ylabel('Current (pA)')
axes[1].set_xlabel('Time (ms)')
plt.subplots_adjust(hspace=0)
plt.savefig('%s/%s-pp-model-only.png' % (savedir, saveas), dpi=200,
        bbox_inches='tight')
plt.close()

# GP only
gp_only_mean = np.mean(gp_only_ppc_mean, axis=0)
var1 = np.mean(gp_ppc_var, axis=0)
var2_1 = np.mean(np.power(gp_only_ppc_mean, 2), axis=0)
var2_2 = np.power(np.mean(gp_only_ppc_mean, axis=0), 2)
gp_only_sd = np.sqrt(var1 + var2_1 - var2_2)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
        gridspec_kw={'height_ratios': [1, 2]})
axes[0].plot(times, voltage, c='#7f7f7f')
axes[0].set_ylabel('Voltage (mV)')
axes[1].plot(times, data - model_mean, alpha=0.5, label='data')
axes[1].plot(times, gp_only_mean, label='Mean')
n_sd = scipy_stats_norm.ppf(1. - .05 / 2.)
axes[1].plot(times, gp_only_mean + n_sd * gp_only_sd, '-', color='blue',
        lw=0.5, label='95% C.I.')
axes[1].plot(times, gp_only_mean - n_sd * gp_only_sd, '-', color='blue',
        lw=0.5)
axes[1].legend()
axes[1].set_ylabel('Current (pA)')
axes[1].set_xlabel('Time (ms)')
plt.subplots_adjust(hspace=0)
plt.savefig('%s/%s-pp-gp-only.png' % (savedir, saveas), dpi=200,
        bbox_inches='tight')
plt.close()

# Save as text
np.savetxt('%s/raw/%s-pp-time.txt' % (savedir, saveas), times)
np.savetxt('%s/raw/%s-pp-gp-mean.txt' % (savedir, saveas), ppc_mean)
np.savetxt('%s/raw/%s-pp-gp-sd.txt' % (savedir, saveas), ppc_sd)
np.savetxt('%s/raw/%s-pp-only-model-mean.txt' % (savedir, saveas), model_mean)
np.savetxt('%s/raw/%s-pp-only-model-sd.txt' % (savedir, saveas), model_sd)
np.savetxt('%s/raw/%s-pp-only-gp-mean.txt' % (savedir, saveas), gp_only_mean)
np.savetxt('%s/raw/%s-pp-only-gp-sd.txt' % (savedir, saveas), gp_only_sd)
