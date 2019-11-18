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
from sparse_gp_custom_likelihood import _create_theano_conditional_graph

import model as m

"""
Posterior predictive with i.i.d. noise model.
"""

def rmse(t1, t2):
    # Root mean square error
    return np.sqrt(np.mean(np.power(np.subtract(t1, t2), 2)))

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

savedir = './fig/mcmc-' + info_id
if not os.path.isdir(savedir):
    os.makedirs(savedir)
if not os.path.isdir(savedir + '/raw'):
    os.makedirs(savedir + '/raw')

data_file_name = 'data-%s.csv' % which_predict
print('Predicting ', data_file_name)
saveas = info_id + '-sinewave-' + which_predict

loaddir = './out/mcmc-' + info_id
loadas = info_id + '-sinewave'

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

# Load model
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
loglikelihood = pints.GaussianLogLikelihood(problem)
logmodelprior = LogPrior[info_id](transform_to_model_param,
        transform_from_model_param)
lognoiseprior = pints.UniformLogPrior([0.1 * noise_sigma], [10. * noise_sigma])
logprior = pints.ComposedLogPrior(logmodelprior, lognoiseprior)
logposterior = pints.LogPosterior(loglikelihood, logprior)

# Load MCMC results
ppc_samples = pints.io.load_samples('%s/%s-chain_0.csv' % (loaddir, loadas))

# Compute
ppc_size = np.size(ppc_samples, axis=0)
ppc = []
model_rmse = []
posterior_all = []

for ind in np.random.choice(range(0, ppc_size), 100, replace=False):
    # Expecting these parameters can be used for simulation
    params = ppc_samples[ind, :]

    # Simulate
    current_valid_protocol = model.simulate(params, times)

    ppc.append(current_valid_protocol)

    # To compute E[rmse]
    model_rmse.append(rmse(data, current_valid_protocol))

    # To compute E[posterior]
    posterior_all.append(logposterior(params))

# Compute E[rmse]
expected_model_rmse = np.mean(model_rmse, axis=0)
np.savetxt('%s/%s-rmse.txt' % (savedir, saveas), [expected_model_rmse])

# Compute E[posterior]
expected_posterior = np.mean(posterior_all, axis=0)
np.savetxt('%s/%s-posterior.txt' % (savedir, saveas), [expected_posterior])

n_sd = scipy_stats_norm.ppf(1. - .05 / 2.)
ppc_mean = np.mean(ppc, axis=0)
var_1 = np.mean(np.power(ppc, 2), axis=0)
var_2 = np.power(np.mean(ppc, axis=0), 2)
ppc_sd = np.sqrt(var_1 - var_2)
print(np.sum(np.abs(ppc_sd - np.std(ppc, axis=0))))

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

# Save as text
np.savetxt('%s/raw/%s-pp-time.txt' % (savedir, saveas), times)
np.savetxt('%s/raw/%s-pp-mean.txt' % (savedir, saveas), ppc_mean)
np.savetxt('%s/raw/%s-pp-sd.txt' % (savedir, saveas), ppc_sd)
