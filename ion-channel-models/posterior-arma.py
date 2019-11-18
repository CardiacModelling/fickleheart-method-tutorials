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
from scipy.stats import norm as scipy_stats_norm

import model as m

"""
Posterior predictive with ARMA noise model.
"""

def rmse(t1, t2):
    # Root mean square error
    return np.sqrt(np.mean(np.power(np.subtract(t1, t2), 2)))

model_list = ['A', 'B', 'C']
predict_list = ['sinewave', 'staircase', 'activation', 'ap']

np.random.seed(101)  # fix seed for prediction

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
if not os.path.isdir(savedir + '/raw'):
    os.makedirs(savedir + '/raw')

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

# Simulate voltage
voltage = model.voltage(times)

# Create posterior
import importlib
sys.path.append('./mmt-model-files')
info_id = 'model_%s' % which_model
info = importlib.import_module(info_id)
import parametertransform
transform_to_model_param = parametertransform.donothing
transform_from_model_param = parametertransform.donothing
noise_sigma = np.std(data[:500])
import priors
LogPrior = {
        'model_A': priors.ModelALogPrior,
        'model_B': priors.ModelBLogPrior,
        }
problem = pints.SingleOutputProblem(model, times, data)
from armax_ode_tsa_likelihood import DiscrepancyLogLikelihood
transparams = False
loglikelihood = DiscrepancyLogLikelihood(problem, armax_result, transparams=transparams)
logmodelprior = LogPrior[info_id](transform_to_model_param,
        transform_from_model_param)
from priors import ArmaNormalCentredLogPrior
logarmaprior = ArmaNormalCentredLogPrior(armax_result, 0.25)
# Compose all priors
logprior = pints.ComposedLogPrior(logmodelprior, logarmaprior)
logposterior = pints.LogPosterior(loglikelihood, logprior)

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
model_mean = []
armax_only_mean = []
armax_only_sd = []
armax_rmse = []
model_rmse = []
posterior_all = []

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
    ao_mean, ao_sd, _ = armax_result.forecast(steps=len(times), exog=np.zeros(len(ode_sol)))
    armax_result.model.exog = ode_sol[:, None]

    armax_mean.append(mean)
    armax_sd.append(sd)
    model_mean.append(ode_sol)
    #armax_only_mean.append(mean - ode_sol)
    armax_only_mean.append(ao_mean)
    armax_only_sd.append(ao_sd)

    # To compute E[rmse]
    ppc_sample_sample = scipy_stats_norm(mean, sd).rvs()
    armax_rmse.append(rmse(data, ppc_sample_sample))
    model_rmse.append(rmse(data, ode_sol))

    # To compute E[posterior]
    params = np.copy(ppc_samples[ind, :])
    posterior_all.append(logposterior(params))

# Compute E[rmse]
expected_armax_rmse = np.mean(armax_rmse, axis=0)
expected_model_rmse = np.mean(model_rmse, axis=0)
np.savetxt('%s/%s-armax-rmse.txt' % (savedir, saveas), [expected_armax_rmse])
np.savetxt('%s/%s-model-rmse.txt' % (savedir, saveas), [expected_model_rmse])

# Compute E[posterior]
expected_posterior = np.mean(posterior_all, axis=0)
np.savetxt('%s/%s-posterior.txt' % (savedir, saveas), [expected_posterior])

n_sd = scipy_stats_norm.ppf(1. - .05 / 2.)

# Model + ARMAX
ppc_mean = np.mean(armax_mean, axis=0)
var1 = np.mean(np.power(armax_sd, 2), axis=0)
var2_1 = np.mean(np.power(armax_mean, 2), axis=0)
var2_2 = np.power(np.mean(armax_mean, axis=0), 2)
ppc_sd = np.sqrt(var1 + (var2_1 - var2_2))

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
axes[0].set_title('ODE model + ARMA(%s, %s)' % (arma_p, arma_q))
plt.subplots_adjust(hspace=0)
plt.savefig('%s/%s-pp.png' % (savedir, saveas), dpi=200,
        bbox_inches='tight')
plt.close()

# Model only
model_ppc_mean = np.mean(model_mean, axis=0)
var1_1 = np.mean(np.power(model_mean, 2), axis=0)
var1_2 = np.power(np.mean(model_mean, axis=0), 2)
model_ppc_sd = np.sqrt(var1_1 - var1_2)
print(np.sum(np.abs(model_ppc_sd - np.std(model_mean, axis=0))))

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
        gridspec_kw={'height_ratios': [1, 2]})
axes[0].plot(times, voltage, c='#7f7f7f')
axes[0].set_ylabel('Voltage (mV)')
axes[1].plot(times, data, alpha=0.5, label='data')
axes[1].plot(times, model_ppc_mean, label='Mean')
axes[1].plot(times, model_ppc_mean + n_sd * model_ppc_sd, '-', color='blue',
        lw=0.5, label='95% C.I.')
axes[1].plot(times, model_ppc_mean - n_sd * model_ppc_sd, '-', color='blue',
        lw=0.5)
axes[1].legend()
axes[1].set_ylabel('Current (pA)')
axes[1].set_xlabel('Time (ms)')
axes[0].set_title('ODE model only')
plt.subplots_adjust(hspace=0)
plt.savefig('%s/%s-pp-model-only.png' % (savedir, saveas), dpi=200,
        bbox_inches='tight')
plt.close()

# ARMAX only
armax_ppc_mean = np.mean(armax_only_mean, axis=0)
var1 = np.mean(np.power(armax_only_sd, 2), axis=0)
var2_1 = np.mean(np.power(armax_only_mean, 2), axis=0)
var2_2 = np.power(np.mean(armax_only_mean, axis=0), 2)
armax_ppc_sd = np.sqrt(var1 + (var2_1 - var2_2))

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
        gridspec_kw={'height_ratios': [1, 2]})
axes[0].plot(times, voltage, c='#7f7f7f')
axes[0].set_ylabel('Voltage (mV)')
axes[1].plot(times, data - model_ppc_mean, alpha=0.5, label='data')
axes[1].plot(times, armax_ppc_mean, label='Mean')
axes[1].plot(times, armax_ppc_mean + n_sd * armax_ppc_sd, '-', color='blue',
        lw=0.5, label='95% C.I.')
axes[1].plot(times, armax_ppc_mean - n_sd * armax_ppc_sd, '-', color='blue',
        lw=0.5)
axes[1].legend()
axes[1].set_ylabel('Current (pA)')
axes[1].set_xlabel('Time (ms)')
axes[0].set_title('ARMA(%s, %s) only' % (arma_p, arma_q))
plt.subplots_adjust(hspace=0)
plt.savefig('%s/%s-pp-armax-only.png' % (savedir, saveas), dpi=200,
        bbox_inches='tight')
plt.close()

# Save as text
np.savetxt('%s/raw/%s-pp-time.txt' % (savedir, saveas), times)
np.savetxt('%s/raw/%s-pp-armax-mean.txt' % (savedir, saveas), ppc_mean)
np.savetxt('%s/raw/%s-pp-armax-sd.txt' % (savedir, saveas), ppc_sd)
np.savetxt('%s/raw/%s-pp-only-model-mean.txt' % (savedir, saveas), model_ppc_mean)
np.savetxt('%s/raw/%s-pp-only-model-sd.txt' % (savedir, saveas), model_ppc_sd)
np.savetxt('%s/raw/%s-pp-only-armax-mean.txt' % (savedir, saveas), armax_ppc_mean)
np.savetxt('%s/raw/%s-pp-only-armax-sd.txt' % (savedir, saveas), armax_ppc_sd)
