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
import statsmodels.api as sm
import joblib

import model as m

"""
Posterior predictive with ARMA noise model.
"""

model_list = ['A', 'B', 'C']
predict_list = ['sinewave', 'staircase', 'activation', 'ap']

try:
    which_model = sys.argv[1] 
    arma_p = int(sys.argv[2])
    arma_q = int(sys.argv[3])
    which_predict = sys.argv[4]
except:
    print('Usage: python %s [str:which_model]' % os.path.basename(__file__)
            + ' [int:arma_p] [int:arma_q] [str:which_predict]')
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

savedir = './fig/mcmc-' + info_id + '-arma_%s_%s' % (arma_p, arma_q)
if not os.path.isdir(savedir):
    os.makedirs(savedir)

data_file_name = 'data-%s.csv' % which_predict
print('Predicting ', data_file_name)
saveas = info_id + '-sinewave-' + which_predict

loaddir = './out/mcmc-' + info_id + '-arma_%s_%s' % (arma_p, arma_q)
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

# Load model
model = m.Model(info.model_file,
        variables=info.parameters,
        current_readout=info.current_list,
        set_ion=info.ions_conc,
        transform=None,
        temperature=273.15 + info.temperature,  # K
        )

# Update protocol to training protocol
model.set_fixed_form_voltage_protocol(protocol_train, protocol_train_times)

# Load fitting results
calloaddir = './out/' + info_id
load_seed = '542811797'
fit_idx = [1, 2, 3]
model_x0_list = []
for i in fit_idx:
    f = '%s/%s-solution-%s-%s.txt' % (calloaddir, 'sinewave', load_seed, i)
    p = np.loadtxt(f)
    model_x0_list.append(p)

# Fit an armax model to get ballpark estmates of starting arma parameters
cmaes_params = model_x0_list[0]
exog_current = model.simulate(cmaes_params, times_train)[:, None]
armax_result = joblib.load('%s/%s-armax.pkl' % (loaddir, loadas))
n_arama = len(armax_result.params[armax_result.k_exog:])

# Update protocol to predicting protocol
model.set_fixed_form_voltage_protocol(protocol, protocol_times)

# Load MCMC results
ppc_samples = pints.io.load_samples('%s/%s-chain_0.csv' % (loaddir, loadas))


# Bayesian prediction of ARMAX Based on the variance identity
# -----------------------------------------------------------------------------
# That is let say that theta = (ode_params, arma_params) and
# p(theta|data) is the posterior. We want to evaluate the posterior
# predictive: E|armax_forecast|data|, Var|armax_forecast|data|, all
# expectation w.r.t p(theta|data). This can be done with the variance
# identitiy trick.
# -----------------------------------------------------------------------------
ppc_size = np.size(ppc_samples, axis=0)
armax_mean = []
armax_sd = []

for ind in np.random.choice(range(0, ppc_size), 100, replace=False):
    ode_params = np.copy(ppc_samples[ind, :-n_arama])
    ode_sol = model.simulate(ode_params, times)
    armax_params = np.append(1.0, ppc_samples[ind, -n_arama:])
    armax_result.params = armax_params
    armax_result.arparams = armax_params[armax_result.k_exog:\
            armax_result.k_ar + armax_result.k_exog]
    armax_result.maparams = armax_params[-armax_result.k_ma:]
    armax_result.model.exog = exog_current  # 'old sim.' from training prtocol
    mean, sd, _ = armax_result.forecast(steps=len(times), exog=ode_sol)
    armax_result.model.exog = ode_sol[:, None]
    armax_mean.append(mean)
    armax_sd.append(sd)

armax_mean = np.array(armax_mean)
armax_sd = np.array(armax_sd)
ppc_mean = np.mean(armax_mean, axis=0)
var1 = np.mean(armax_sd**2, axis=0)
var2 = np.mean(armax_mean**2, axis=0)
var3 = (np.mean(armax_mean, axis=0))**2
ppc_sd = np.sqrt(var1 + var2 + var3)

plt.figure(figsize=(8, 6))
plt.plot(times, data, label='Data')
plt.plot(times, ppc_mean, label='Mean')
plt.plot(times, ppc_mean + 2 * ppc_sd, '-', color='blue', lw=0.5,
        label='conf_int')
plt.plot(times, ppc_mean - 2 * ppc_sd, '-', color='blue', lw=0.5)
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Current (pA)')
plt.savefig('%s/%s-pp.png' % (savedir, saveas))
plt.close()

