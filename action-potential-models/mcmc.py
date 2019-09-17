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
import parametertransform
import protocol

"""
Run MCMC assuming iid Gaussian noise.
"""

model_list = ['tnnp-2004-w', 'fink-2008']
data_list = ['stim1hz', 'stim2hz', 'randstim']

try:
    which_model = sys.argv[1]
    which_data = sys.argv[2]
except:
    print('Usage: python %s [str:which_model]' % os.path.basename(__file__)
            + ' [str:which_data]')
    sys.exit()

if which_model not in model_list:
    raise ValueError('Input model %s is not available in the model list' \
            % which_model)

if which_data not in data_list:
    raise ValueError('Input data %s is not available in the data list' \
            % which_data)

data_dir = './data'

savedir = './out/mcmc-' + which_model
if not os.path.isdir(savedir):
    os.makedirs(savedir)

data_file_name = 'data-%s.csv' % which_data
print('Fitting to ', data_file_name)
saveas = which_model + '-' + which_data

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
loglikelihood = pints.GaussianLogLikelihood(problem)
logprior = pints.UniformLogPrior(
        np.append(np.log(0.1) * np.ones(model.n_parameters()),
            0.1 * noise_sigma),
        np.append(np.log(10.) * np.ones(model.n_parameters()),
            10. * noise_sigma)
        )
logposterior = pints.LogPosterior(loglikelihood, logprior)

# Check logposterior is working fine
priorparams = np.ones(model.n_parameters())
transform_priorparams = transform_from_model_param(priorparams)
priorparams = np.append(priorparams, noise_sigma)
transform_priorparams = np.append(transform_priorparams, noise_sigma)
print('Score at prior parameters: ',
        logposterior(transform_priorparams))
for _ in range(10):
    assert(logposterior(transform_priorparams) ==\
            logposterior(transform_priorparams))

# Load fitting results
calloaddir = './out/' + which_model
load_seed = 542811797
fit_idx = [1, 2, 3]
transform_x0_list = []
print('MCMC starting point: ')
for i in fit_idx:
    f = '%s/%s-solution-%s-%s.txt' % (calloaddir, which_data, load_seed, i)
    p = np.loadtxt(f)
    transform_x0_list.append(np.append(transform_from_model_param(p),
            noise_sigma))
    print(transform_x0_list[-1])

# Run
mcmc = pints.MCMCController(logposterior, 3, transform_x0_list,
        method=pints.PopulationMCMC)
n_iter = 100000
mcmc.set_max_iterations(n_iter)
mcmc.set_initial_phase_iterations(int(0.05 * n_iter))
mcmc.set_parallel(False)
mcmc.set_chain_filename('%s/%s-chain.csv' % (savedir, saveas))
mcmc.set_log_pdf_filename('%s/%s-pdf.csv' % (savedir, saveas))
chains = mcmc.run()

# De-transform parameters
chains_param = np.zeros(chains.shape)
for i, c in enumerate(chains):
    c_tmp = np.copy(c)
    chains_param[i, :, :-1] = transform_to_model_param(c_tmp[:, :-1])
    chains_param[i, :, -1] = c_tmp[:, -1]
    del(c_tmp)

# Save (de-transformed version)
pints.io.save_samples('%s/%s-chain.csv' % (savedir, saveas), *chains_param)

# Plot
# burn in and thinning
chains_final = chains[:, int(0.5 * n_iter)::5, :]
chains_param = chains_param[:, int(0.5 * n_iter)::5, :]

transform_x0 = transform_x0_list[0]
x0 = np.append(transform_to_model_param(transform_x0[:-1]), transform_x0[-1])

pints.plot.pairwise(chains_param[0], kde=False, ref_parameters=x0)
plt.savefig('%s/%s-fig1.png' % (savedir, saveas))
plt.close('all')

pints.plot.trace(chains_param, ref_parameters=x0)
plt.savefig('%s/%s-fig2.png' % (savedir, saveas))
plt.close('all')

pints.plot.trace(chains_final, ref_parameters=transform_x0)
plt.savefig('%s/%s-fig2-transformed.png' % (savedir, saveas))
plt.close('all')

pints.plot.series(chains_final[0], problem)  # use search space parameters
plt.savefig('%s/%s-fig3.png' % (savedir, saveas))
plt.close('all')

# Check convergence using rhat criterion
print('R-hat:')
print(pints.rhat_all_params(chains_param))

print('Done.')
#eof
