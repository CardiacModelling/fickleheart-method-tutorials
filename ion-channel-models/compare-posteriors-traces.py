#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('./method')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import pints.io
import pints.plot

"""
Posterior distributions with different discrepancy models.
"""

model_list = ['A', 'B', 'C']
discrepancy_list = ['', '-gp', '-gp-ov', '-arma_2_2']
discrepancy_names = ['iid noise', 'GP(t)', 'GP(O, V)', 'ARMA(2, 2)']

try:
    which_model = sys.argv[1] 
    which_discrepancy = int(sys.argv[2])
except:
    print('Usage: python %s [str:which_model]' % os.path.basename(__file__)
            + ' [str:which_discrepancy]')
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

savedir = './fig'
if not os.path.isdir(savedir):
    os.makedirs(savedir)
saveas = info_id + '-sinewave-posteriors-traces' \
        + discrepancy_list[which_discrepancy]

# Load MCMC results
lastniter = 25000
thinning = 25
loaddir = './out/mcmc-' + info_id + discrepancy_list[which_discrepancy]
loadas = info_id + '-sinewave'
samples = pints.io.load_samples('%s/%s-chain.csv'
        % (loaddir, loadas), 3)
# burn in and thinning
samples = [x[-lastniter::thinning, :] for x in samples]

# Traces
bins = 40
alpha = 0.5
n_percentiles = None
if which_model == 'A':
    fig, axes = plt.subplots(int(np.ceil(n_parameters / 3)), 3,
            figsize=(10, 5))
else:
    fig, axes = plt.subplots(int(np.ceil(n_parameters / 3)), 3,
            figsize=(10, 9))
plt.subplots_adjust(hspace=.6, wspace=.3)
for i in range(axes.size):
    ai, aj = int(i // 3), i % 3
    if not (i < n_parameters):
        axes[ai, aj].axis('off')
        continue
    if which_model == 'A':
        axes[ai, aj].set_ylabel(info.parameters_nice[i], fontsize=14)
        if ai == 2:
            axes[ai, aj].set_xlabel('#Iteration', fontsize=14)
    else:
        raise NotImplementedError #TODO
        axes[ai, aj].text(0.5, -0.3, info.parameters_nice[i], fontsize=14,
                ha='center', va='center', transform=axes[ai, aj].transAxes)
        if aj == 0:
            axes[ai, aj].text(-0.3, 0.5, 'Marginal\nposterior', fontsize=14,
                    ha='center', va='center', transform=axes[ai, aj].transAxes,
                    rotation=90)
    axes[ai, aj].ticklabel_format(axis='both', style='sci', scilimits=(-2, 3))
    for j, samples_j in enumerate(samples):
        xmin = np.min(samples_j[:, i])
        xmax = np.max(samples_j[:, i])
        xbins = np.linspace(xmin, xmax, bins)
        axes[ai, aj].plot(samples_j[:, i], alpha=alpha,
                label='Chain #%s' % (j + 1),
                color='C' + str(j))
axes[0, 2].legend(loc='lower right', bbox_to_anchor=(1, 1.1), ncol=4,
        bbox_transform=axes[0, 2].transAxes)
axes[0, 0].text(0, 1.2,
        'Discrepancy model: ' + discrepancy_names[which_discrepancy],
        fontsize=14, ha='left', va='bottom', transform=axes[0, 0].transAxes)
plt.savefig('%s/%s' % (savedir, saveas), dpi=200, bbox_inches='tight')
plt.close('all')

