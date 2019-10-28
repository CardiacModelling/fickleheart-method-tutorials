#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('./method')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pints.io
import pints.plot

"""
Posterior distributions with different discrepancy models.
"""

model_list = ['A', 'B', 'C']
discrepancy_list = ['', '-gp']#, '-arma_2_2']
discrepancy_names = ['iid noise', 'GP']#, 'ARMAX(2, 2)']

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
n_parameters = len(info.parameters)

savedir = './fig/compare'
if not os.path.isdir(savedir):
    os.makedirs(savedir)
saveas = 'compare-' + info_id + '-sinewave-posteriors'

# Load MCMC results
all_samples = []
lastniter = 25000
thinning = 5
for d in discrepancy_list:
    loaddir = './out/mcmc-' + info_id + d
    loadas = info_id + '-sinewave'
    samples = pints.io.load_samples('%s/%s-chain_2.csv' % (loaddir, loadas))
    # burn in and thinning
    samples = samples[-lastniter::thinning, :]
    all_samples.append(samples)

# Histograms
bins = 40
alpha = 0.5
n_percentiles = None
fig, axes = plt.subplots(int(np.ceil(n_parameters / 3)), 3, figsize=(10, 7))
for i in range(axes.size):
    ai, aj = int(i // 3), i % 3
    if not (i < n_parameters):
        axes[ai, aj].axis('off')
        continue
    axes[ai, aj].set_xlabel(info.parameters_nice[i], fontsize=14)
    if aj == 0:
        axes[ai, aj].set_ylabel('Marginal\nposterior', fontsize=14)
    axes[ai, aj].ticklabel_format(axis='both', style='sci', scilimits=(-2, 3))
    for j, samples_j in enumerate(all_samples):
        if n_percentiles is None:
            xmin = np.min(samples_j[:, i])
            xmax = np.max(samples_j[:, i])
        else:
            xmin = np.percentile(samples_j[:, i],
                                 50 - n_percentiles / 2.)
            xmax = np.percentile(samples_j[:, i],
                                 50 + n_percentiles / 2.)
        xbins = np.linspace(xmin, xmax, bins)
        axes[ai, aj].hist(samples_j[:, i], bins=xbins, alpha=alpha,
                density=True, label=discrepancy_names[j], color='C' + str(j))
axes[0, 0].legend()
#plt.subplots_adjust(hspace=0., wspace=0.)
plt.tight_layout()
plt.savefig('%s/%s' % (savedir, saveas), dpi=200, bbox_inches='tight')
plt.close('all')

