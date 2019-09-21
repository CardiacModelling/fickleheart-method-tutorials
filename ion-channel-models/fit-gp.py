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

import model as m
import parametertransform
import priors
from priors import HalfNormalLogPrior, InverseGammaLogPrior
from sparse_gp_custom_likelihood import DiscrepancyLogLikelihood
"""
Run fit.
"""

model_list = ['A', 'B', 'C']

try:
    which_model = sys.argv[1]
except:
    print('Usage: python %s [str:which_model]' % os.path.basename(__file__)
          + ' --optional [N_repeats]')
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

savedir = './out/' + info_id + '-gp'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

data_file_name = 'data-sinewave.csv'
print('Fitting to ', data_file_name)
print('Temperature: ', info.temperature)
saveas = data_file_name[5:][:-4]

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
noise_sigma = np.std(data[:500])
print('Estimated noise level: ', noise_sigma)

# Model
model = m.Model(info.model_file,
        variables=info.parameters,
        current_readout=info.current_list,
        set_ion=info.ions_conc,
        transform=transform_to_model_param,
        temperature=273.15 + info.temperature,  # K
        )

LogPrior = {
        'model_A': priors.ModelALogPrior,
        'model_B': priors.ModelBLogPrior,
        }

# Update protocol
model.set_fixed_form_voltage_protocol(protocol, protocol_times)

# Create Pints stuffs
inducing_times = times[::1000] # inducing or speudo points for the FITC GP
problem = pints.SingleOutputProblem(model, times, data)
loglikelihood = DiscrepancyLogLikelihood(problem, inducing_times,
        downsample=None)
logmodelprior = LogPrior[info_id](transform_to_model_param,
        transform_from_model_param)
# Priors for discrepancy
# This will have considerable mass at the initial value
lognoiseprior = HalfNormalLogPrior(sd=25, transform=True)
logrhoprior = InverseGammaLogPrior(alpha=5, beta=5, transform=True)
logkersdprior = InverseGammaLogPrior(alpha=5, beta=5, transform=True)
# Compose all priors
logprior = pints.ComposedLogPrior(logmodelprior, lognoiseprior, logrhoprior,
        logkersdprior)
logposterior = pints.LogPosterior(loglikelihood, logprior)

# Check logposterior is working fine
priorparams = np.copy(info.base_param)
transform_priorparams = transform_from_model_param(priorparams)
# Stack non-model parameters together
initial_rho = 0.5  # Kernel hyperparameter \rho
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

# Run
try:
    N = int(sys.argv[2])
except IndexError:
    N = 3

params, logposteriors = [], []

for i in range(N):

    if i == 0:
        x0 = transform_priorparams
    else:
        # Randomly pick a starting point
        x0 = logprior.sample(n=1)[0]
    print('Starting point: ', x0)

    # Create optimiser
    print('Starting logposterior: ', logposterior(x0))
    opt = pints.OptimisationController(logposterior, x0, method=pints.CMAES)
    opt.set_max_iterations(None)
    opt.set_parallel(True)

    # Run optimisation
    try:
        with np.errstate(all='ignore'):
            # Tell numpy not to issue warnings
            p, s = opt.run()
            p[:-3] = transform_to_model_param(p[:-3])
            p[-3:] = np.exp(p[-3:])  # non-model parameter transformation
            params.append(p)
            logposteriors.append(s)
            print('Found solution:          Old parameters:' )
            for k, x in enumerate(p):
                print(pints.strfloat(x) + '    ' + \
                        pints.strfloat(priorparams[k]))
    except ValueError:
        import traceback
        traceback.print_exc()

# Order from best to worst
order = np.argsort(logposteriors)[::-1]  # (use [::-1] for LL)
logposteriors = np.asarray(logposteriors)[order]
params = np.asarray(params)[order]

# Show results
bestn = min(3, N)
print('Best %d logposteriors:' % bestn)
for i in range(bestn):
    print(logposteriors[i])
print('Mean & std of logposterior:')
print(np.mean(logposteriors))
print(np.std(logposteriors))
print('Worst logposterior:')
print(logposteriors[-1])

# Extract best 3
obtained_logposterior0 = logposteriors[0]
obtained_parameters0 = params[0]
obtained_logposterior1 = logposteriors[1]
obtained_parameters1 = params[1]
obtained_logposterior2 = logposteriors[2]
obtained_parameters2 = params[2]

# Show results
print('Found solution:          Old parameters:' )
# Store output
with open('%s/%s-solution-%s-1.txt' % (savedir, saveas, fit_seed), 'w') as f:
    for k, x in enumerate(obtained_parameters0):
        print(pints.strfloat(x) + '    ' + pints.strfloat(priorparams[k]))
        f.write(pints.strfloat(x) + '\n')
print('Found solution:          Old parameters:' )
# Store output
with open('%s/%s-solution-%s-2.txt' % (savedir, saveas, fit_seed), 'w') as f:
    for k, x in enumerate(obtained_parameters1):
        print(pints.strfloat(x) + '    ' + pints.strfloat(priorparams[k]))
        f.write(pints.strfloat(x) + '\n')
print('Found solution:          Old parameters:' )
# Store output
with open('%s/%s-solution-%s-3.txt' % (savedir, saveas, fit_seed), 'w') as f:
    for k, x in enumerate(obtained_parameters2):
        print(pints.strfloat(x) + '    ' + pints.strfloat(priorparams[k]))
        f.write(pints.strfloat(x) + '\n')

