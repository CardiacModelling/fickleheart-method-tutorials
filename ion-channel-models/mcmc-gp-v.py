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
from scipy.interpolate import interp1d

import model as m
import parametertransform
import priors
from priors import HalfNormalLogPrior, InverseGammaLogPrior
from sparse_gp_custom_likelihood_new import DiscrepancyLogLikelihood

"""
Run MCMC.
"""

model_list = ['A', 'B', 'C']

try:
    which_model = sys.argv[1] 
except:
    print('Usage: python %s [str:which_model]' % os.path.basename(__file__))
    sys.exit()

if which_model not in model_list:
    raise ValueError('Input model %s is not available in the model list' \
            % which_model)

# Get all input variables
import importlib
sys.path.append('./mmt-model-files')
info_id = 'model_%s' % which_model
info = importlib.import_module(info_id)

data_dir = './data'

savedir = './out/mcmc-' + info_id + '-gp-v'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

data_file_name = 'data-sinewave.csv'
print('Fitting to ', data_file_name)
print('Temperature: ', info.temperature)
saveas = info_id + '-' + data_file_name[5:][:-4]

# Protocol
protocol = np.loadtxt('./protocol-time-series/sinewave.csv', skiprows=1,
        delimiter=',')
protocol_times = protocol[:, 0]
protocol = protocol[:, 1]


# Control fitting seed
# fit_seed = np.random.randint(0, 2**30)
fit_seed = 542811797
print('Fit seed: ', fit_seed)

np.random.seed(fit_seed)

# Set parameter transformation
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param

# Load data
data = np.loadtxt(data_dir + '/' + data_file_name,
                  delimiter=',', skiprows=1)  # headers
times = data[:, 0]
data = data[:, 1]
voltage = interp1d(protocol_times, protocol, kind='linear')(times)
noise_sigma = np.std(data[:500])
print('Estimated noise level: ', noise_sigma)

model = m.Model(info.model_file,
        variables=info.parameters,
        current_readout=info.current_list,
        set_ion=info.ions_conc,
        transform=transform_to_model_param,
        temperature=273.15 + info.temperature,  # K
        )
model.return_open(return_open=False)  # set simulate() to return current only

LogPrior = {
        'model_A': priors.ModelALogPrior,
        'model_B': priors.ModelBLogPrior,
        }

# Update protocol
model.set_fixed_form_voltage_protocol(protocol, protocol_times)

# Create Pints stuffs
USE_PROBABILITY_WITH_VOLTAGE = False
NUM_IND_THIN = 1000
problem = pints.SingleOutputProblem(model, times, data)
loglikelihood = DiscrepancyLogLikelihood(problem, voltage=voltage,
        num_ind_thin=NUM_IND_THIN, use_open_prob=USE_PROBABILITY_WITH_VOLTAGE,
        downsample=None)
logmodelprior = LogPrior[info_id](transform_to_model_param,
        transform_from_model_param)
# Priors for discrepancy
# This will have considerable mass at the initial value
lognoiseprior = HalfNormalLogPrior(sd=25, transform=True)
if USE_PROBABILITY_WITH_VOLTAGE:
    logrhoprior1 = InverseGammaLogPrior(alpha=5,beta=5,transform=True) 
    logrhoprior2 = InverseGammaLogPrior(alpha=5,beta=5,transform=True) 
    logrhoprior = pints.ComposedLogPrior(logrhoprior1, logrhoprior2) 
    nds = 4  # Number of non-model parameters
else:
    logrhoprior = InverseGammaLogPrior(alpha=5,beta=5,transform=True)
    nds = 3  # Number of non-model parameters
logkersdprior = InverseGammaLogPrior(alpha=5, beta=5, transform=True)
# Compose all priors
logprior = pints.ComposedLogPrior(logmodelprior, lognoiseprior, logrhoprior,
        logkersdprior)
logposterior = pints.LogPosterior(loglikelihood, logprior)

# Check logposterior is working fine
priorparams = np.copy(info.base_param)
transform_priorparams = transform_from_model_param(priorparams)
# Stack non-model parameters together
if USE_PROBABILITY_WITH_VOLTAGE:
    initial_rho = [0.5, 0.5]  # Kernel hyperparameter \rho
else:
    initial_rho = 0.5
initial_ker_sigma = 5.0  # Kernel hyperparameter \ker_sigma
priorparams = np.hstack((priorparams, noise_sigma, initial_rho,
        initial_ker_sigma))
transform_priorparams = np.hstack((transform_priorparams, np.log(noise_sigma),
        np.log(initial_rho), np.log(initial_ker_sigma)))
print('Posterior at prior parameters: ',
        logposterior(transform_priorparams))
for _ in range(10):
    assert(logposterior(transform_priorparams) ==\
            logposterior(transform_priorparams))

# Load fitting results
calloaddir = './out/' + info_id + '-gp-v'
load_seed = '542811797'
fit_idx = [1, 2, 3]
transform_x0_list = []

print('MCMC starting point: ')
for i in fit_idx:
    f = '%s/%s-solution-%s-%s.txt' % (calloaddir, 'sinewave', load_seed, i)
    p = np.loadtxt(f)
    transform_x0_list.append(np.hstack((
                    transform_from_model_param(p[:-nds]),
                    np.log(p[-nds:]))))
    print(transform_x0_list[-1])
    print('Posterior: ', logposterior(transform_x0_list[-1]))

# Run
mcmc = pints.MCMCController(logposterior, len(transform_x0_list),
        transform_x0_list, method=pints.AdaptiveCovarianceMCMC)
n_iter = 100000
mcmc.set_max_iterations(n_iter)
mcmc.set_initial_phase_iterations(200)  # max 200 iterations for random walk
mcmc.set_parallel(True)
mcmc.set_chain_filename('%s/%s-chain.csv' % (savedir, saveas))
mcmc.set_log_pdf_filename('%s/%s-pdf.csv' % (savedir, saveas))
chains = mcmc.run()

# De-transform parameters
chains_param = np.zeros(chains.shape)
for i, c in enumerate(chains):
    c_tmp = np.copy(c)
    # First the model ones
    chains_param[i, :, :-nds] = transform_to_model_param(
            c_tmp[:, :-nds])
    # Then the discrepancy ones
    chains_param[i, :, -nds:] = np.exp((c_tmp[:, -nds:]))
    del(c_tmp)

# Save (de-transformed version)
pints.io.save_samples('%s/%s-chain.csv' % (savedir, saveas), *chains_param)

# Plot
# burn in and thinning
chains_final = chains[:, int(0.5 * n_iter)::5, :]
chains_param = chains_param[:, int(0.5 * n_iter)::5, :]

transform_x0 = transform_x0_list[0]
x0 = np.append(transform_to_model_param(transform_x0[:-nds]),
        np.exp(transform_x0[-nds:]))

pints.plot.pairwise(chains_param[0], kde=False, ref_parameters=x0)
plt.savefig('%s/%s-fig1.png' % (savedir, saveas))
plt.close('all')

pints.plot.trace(chains_param, ref_parameters=x0)
plt.savefig('%s/%s-fig2.png' % (savedir, saveas))
plt.close('all')

