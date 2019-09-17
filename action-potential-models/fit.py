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
import protocol

"""
Run fit.
"""

model_list = ['tnnp-2004-w', 'fink-2008']
data_list = ['stim1hz', 'stim2hz', 'randstim']

try:
    which_model = sys.argv[1]
    which_data = sys.argv[2]
except:
    print('Usage: python %s [str:which_model]' % os.path.basename(__file__)
            + ' [str:which_data] --optional [N_repeats]')
    sys.exit()

if which_model not in model_list:
    raise ValueError('Input model %s is not available in the model list' \
            % which_model)

if which_data not in data_list:
    raise ValueError('Input data %s is not available in the data list' \
            % which_data)

data_dir = './data'

savedir = './out/' + which_model
if not os.path.isdir(savedir):
    os.makedirs(savedir)

data_file_name = 'data-%s.csv' % which_data
print('Fitting to ', data_file_name)
saveas = which_data

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
noise_sigma = np.std(data[:200])
print('Estimated noise level: ', noise_sigma)

# Protocol
stim_list = {
        'stim1hz': protocol.stim1hz,
        'stim2hz': protocol.stim2hz,
        'randstim': protocol.randstim,
        }
stim_seq = stim_list[which_data]


# Model
model = m.Model(
        './mmt-model-files/%s.mmt' % which_model,
        stim_seq=stim_seq,
        transform=transform_to_model_param,
        )
model.set_name(which_model)

# Create Pints stuffs
problem = pints.SingleOutputProblem(model, times, data)
loglikelihood = pints.GaussianKnownSigmaLogLikelihood(problem, noise_sigma)
logprior = pints.UniformLogPrior(
        np.log(0.1) * np.ones(model.n_parameters()),
        np.log(10.) * np.ones(model.n_parameters())
        )
logposterior = pints.LogPosterior(loglikelihood, logprior)

# Check logposterior is working fine
priorparams = np.ones(model.n_parameters())
transform_priorparams = transform_from_model_param(priorparams)
print('Score at prior parameters: ',
        logposterior(transform_priorparams))
for _ in range(10):
    assert(logposterior(transform_priorparams) ==\
            logposterior(transform_priorparams))

# Run
try:
    N = int(sys.argv[3])
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
    opt = pints.OptimisationController(logposterior, x0.T, method=pints.CMAES)
    opt.set_max_iterations(None)
    opt.set_parallel(True)

    # Run optimisation
    try:
        with np.errstate(all='ignore'):
            # Tell numpy not to issue warnings
            p, s = opt.run()
            p = transform_to_model_param(p)
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
for i in xrange(bestn):
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

fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 4))
sol0 = problem.evaluate(transform_from_model_param(obtained_parameters0))
sol1 = problem.evaluate(transform_from_model_param(obtained_parameters1))
sol2 = problem.evaluate(transform_from_model_param(obtained_parameters2))
axes.plot(times, data, alpha=0.5, label='data')
axes.plot(times, sol0, label='found solution 1')
axes.plot(times, sol1, label='found solution 2')
axes.plot(times, sol2, label='found solution 3')
axes.legend()
axes.set_ylabel('Voltage (mV)')
axes.set_xlabel('Time (ms)')
plt.subplots_adjust(hspace=0)
plt.savefig('%s/%s-solution-%s.png' % (savedir, saveas, fit_seed),
        bbox_inches='tight')
plt.close()

