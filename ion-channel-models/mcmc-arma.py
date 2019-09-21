#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('./method')
import os
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pints
import pints.io
import pints.plot
import pymc3 as pm
import statsmodels.api as sm
import random 
from sklearn.externals import joblib
from statsmodels.tsa.arima_process import arma2ma

import model as m
import parametertransform
import priors
from priors import HalfNormalLogPrior, InverseGammaLogPrior, ArmaLogPrior
from armax_ode_tsa_likelihood import DiscrepancyLogLikelihood
"""
Run fit.

Note for Chon: Here I am using ARMA(2,2) but we need to fit ARMA(1,2), ARMA(2,1), ARMA(1,1) as well
and compare there Deviance information criteria, which this script calculates at the end

"""
print('Using PyMC3 version: ',str(pm.__version__))
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

savedir = './out/mcmc-' + info_id
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
noise_sigma = np.log(np.std(data[:500]))

print('Estimated noise level: ', noise_sigma)

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

# Fit an armax model to get ballpark estmates of starting arma parameters
# I have hard-coded to an ARMA(2,2) model, maybe worth using user iputs
debug = False
if not debug:

        print('Fitting an ARMAX (', str(2),',',str(2),') model')
        cmaes_params = np.copy(np.log(info.base_param))
        exog_current = model.simulate(cmaes_params, times)[:,None]
        armax_mod = sm.tsa.ARMA(data, order=(2,2), exog=exog_current)
        armax_result = armax_mod.fit(trend='nc', transparams= True, solver='cg')
        n_arama = len(armax_result.params[armax_result.k_exog:])
        print(armax_result.summary())
        joblib.dump(armax_result, './out/armax.pkl', compress=3)
else:
        cmaes_params = np.copy(np.log(info.base_param))
        exog_current = model.simulate(cmaes_params, times)[:,None]
        armax_result = joblib.load('./out/armax.pkl')
        n_arama = len(armax_result.params[armax_result.k_exog:])


# Create Pints stuffs
problem = pints.SingleOutputProblem(model, times, data)
loglikelihood = DiscrepancyLogLikelihood(problem, armax_result) #ARMAX likelihood 
logmodelprior = LogPrior[info_id](transform_to_model_param,
        transform_from_model_param)

# Priors for discrepancy
logarmaprior = ArmaLogPrior(armax_result, -2, 2, 0.25) # Note for Chon: Worth checking out more wider/narrower priors

 
logprior = pints.ComposedLogPrior(logmodelprior, logarmaprior)
logposterior = pints.LogPosterior(loglikelihood, logprior)

# Check logposterior is working fine
init_arma = armax_result.params[armax_result.k_exog:]

priorparams = np.copy(info.base_param)
transform_priorparams = transform_from_model_param(priorparams)
priorparams = np.append(priorparams, init_arma)
transform_priorparams = np.append(transform_priorparams, init_arma)
print('Posterior at prior parameters: ',
        logposterior(transform_priorparams))
for _ in range(10):
    assert(logposterior(transform_priorparams) ==\
            logposterior(transform_priorparams))

# Load fitting results
calloaddir = './out/' + info_id
load_seed = 542811797
fit_idx = [1, 2, 3]
transform_x0_list = []

print('MCMC starting point: ')
for i in fit_idx:
    f = '%s/%s-solution-%s-%s.txt' % (calloaddir, 'sinewave', load_seed, i)
    p = np.loadtxt(f)
    transform_x0_list.append(np.append(transform_from_model_param(p),
            init_arma))
    print(transform_x0_list[-1])
    print('Posterior: ', logposterior(transform_x0_list[-1]))

# Run
mcmc = pints.MCMCController(logposterior, len(transform_x0_list),
        transform_x0_list, method=pints.AdaptiveCovarianceMCMC)
n_iter = 200000 # Need higher iterations in my experience
mcmc.set_max_iterations(n_iter)
mcmc.set_initial_phase_iterations(int(200)) # Note for Chon: Only use 100/200 iterations maximum for random walk and then switch to Adaptive
mcmc.set_parallel(True)
mcmc.set_chain_filename('%s/%s-chain.csv' % (savedir, saveas))
mcmc.set_log_pdf_filename('%s/%s-pdf.csv' % (savedir, saveas))
chains = mcmc.run()

# De-transform parameters
chains_param = np.zeros(chains.shape)
for i, c in enumerate(chains):
    c_tmp = np.copy(c)
    chains_param[i, :, :-n_arama] = transform_to_model_param(c_tmp[:, :-n_arama]) # First the model ones 
    chains_param[i, :, -n_arama:] = c_tmp[:, -n_arama:] # Then the discrepancy ones
    
    del(c_tmp)

# Save (de-transformed version)
pints.io.save_samples('%s/%s-chain.csv' % (savedir, saveas), *chains_param)

# Plot
# burn in and thinning
chains_final = chains[:, int(0.5 * n_iter)::1, :]
chains_param = chains_param[:, int(0.5 * n_iter)::1, :]

transform_x0 = transform_x0_list[0]
x0 = np.append(transform_to_model_param(transform_x0[:-n_arama]), transform_x0[-n_arama:])

pints.plot.pairwise(chains_param[0], kde=False, ref_parameters=x0)
plt.savefig('%s/%s-fig1.png' % (savedir, saveas))
plt.close('all')

pints.plot.trace(chains_param, ref_parameters=x0)
plt.savefig('%s/%s-fig2.png' % (savedir, saveas))
plt.close('all')

# Bayesian prediction of ARMAX Based on the variance identity
# That is let say that theta = (ode_params,arma_params) and p(theta|data) is the posterior
# We want to evaluate the posterior predictive: E|armax_forecast|data|, Var|armax_forecast|data|, 
# all expectation w.r.t p(theta|data). This can be done with the variance identitiy trick
ppc_samples = chains_param[0]
armax_mean =[]
armax_sd = []
pdic = []

for ind in random.sample(range(0, np.size(ppc_samples, axis=0)), 100):
        ode_params = transform_from_model_param(ppc_samples[ind, :-n_arama])
        ode_sol = model.simulate(ode_params, times)
        armax_params = np.append(1.0,ppc_samples[ind, -n_arama:])
        armax_result.params = armax_params
        armax_result.arparams = armax_params[armax_result.k_exog:armax_result.k_ar + armax_result.k_exog]
        armax_result.maparams = armax_params[-armax_result.k_ma:]
        armax_result.model.exog = exog_current
        mean, sd, _ = armax_result.forecast(steps=len(times),exog=ode_sol)
        armax_result.model.exog = ode_sol[:,None]
        armax_result.model.transparams = True
        ll = armax_result.model.loglike_kalman(armax_params)
        if ll is not np.inf and ll is not np.nan:

                pdic.append(ll)
                armax_mean.append(mean)
                armax_sd.append(sd)

armax_mean = np.array(armax_mean)
armax_sd = np.array(armax_sd)
ppc_mean = np.mean(armax_mean, axis=0)
var1, var2, var3 = np.mean(armax_sd**2, axis=0), np.mean(armax_mean**2, axis=0), (np.mean(armax_mean, axis=0))**2
ppc_sd = np.sqrt(var1 + var2 + var3)

plt.figure(figsize=(8, 6))
plt.plot(times, data, label='Model C')
plt.plot(times, ppc_mean, label='Mean')
plt.plot(times, ppc_mean + 2*ppc_sd, '-', color='blue',
             lw=0.5, label='conf_int')
plt.plot(times, ppc_mean - 2*ppc_sd, '-', color='blue',
             lw=0.5)
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Current (pA)')
plt.show()

# Calculation of DIC

theta_bar = np.mean(ppc_samples,axis=0)
ode_params = transform_from_model_param(theta_bar[:-n_arama])
ode_sol = model.simulate(ode_params, times)
armax_params = np.append(1.0,theta_bar[-n_arama:])

armax_result.model.exog = ode_sol[:,None]
armax_result.model.transparams = True
pdic = np.mean(pdic)
pdic = 2.0*(armax_result.model.loglike_kalman(armax_params) - pdic)
DIC = -2.0 * armax_result.model.loglike_kalman(armax_params) + 2*pdic
print('DIC for ARMAX(2,2): ',DIC)
