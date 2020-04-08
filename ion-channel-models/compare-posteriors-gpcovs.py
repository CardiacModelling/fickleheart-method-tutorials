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
discrepancy_list = ['-gp-ov', '-gp-ov-OU', '-gp-ov-matern32']
discrepancy_names = ['RBF', 'OU', 'Matern32']
chain_to_use = [0, 0, 0]

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
saveas = 'compare-' + info_id + '-sinewave-posteriors-gp-covs'

# Load MCMC results
all_samples = []
lastniter = 25000
thinning = 25
for i, d in enumerate(discrepancy_list):
    loaddir = './out/mcmc-' + info_id + d
    loadas = info_id + '-sinewave'
    loadchain = chain_to_use[i]
    samples = pints.io.load_samples('%s/%s-chain_%s.csv'
            % (loaddir, loadas, loadchain))
    # burn in and thinning
    samples = samples[-lastniter::thinning, :]
    all_samples.append(samples)

subax_setup = [[0, 0, 0.4, 0.25], [0.6, 0, 0.4, 0.25], (0, 0.75, 0.4, 0.25),
            (0.6, 0.75, 0.4, 0.25)]

# Histograms
bins = 40
alpha = 0.5
n_percentiles = None
if which_model == 'A':
    fig, axes = plt.subplots(int(np.ceil(n_parameters / 3)), 3,
            figsize=(10, 5))
else:
    fig, axes = plt.subplots(int(np.ceil(n_parameters / 3)), 3,
            figsize=(10, 9))
plt.subplots_adjust(hspace=.6, wspace=.15)
for i in range(axes.size):
    ai, aj = int(i // 3), i % 3
    if not (i < n_parameters):
        axes[ai, aj].axis('off')
        continue
    if which_model == 'A':
        axes[ai, aj].set_xlabel(info.parameters_nice[i], fontsize=14)
        if aj == 0:
            axes[ai, aj].set_ylabel('Marginal\nposterior', fontsize=14)
    else:
        axes[ai, aj].text(0.5, -0.3, info.parameters_nice[i], fontsize=14,
                ha='center', va='center', transform=axes[ai, aj].transAxes)
        if aj == 0:
            axes[ai, aj].text(-0.3, 0.5, 'Marginal\nposterior', fontsize=14,
                    ha='center', va='center', transform=axes[ai, aj].transAxes,
                    rotation=90)
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
        if which_model == 'B' and i in [1, 5, 6]:
            # Zooms
            x, y, w, h = subax_setup[j]
            xc, yc = x + w, y + h
            dis = axes[ai, aj].transAxes
            x, y = dis.transform((x, y))
            xc, yc = dis.transform((xc, yc))
            inv = fig.transFigure.inverted()
            xn, yn = inv.transform((x, y))
            xc, yc = inv.transform((xc, yc))
            wn = xc - xn
            hn = yc - yn
            ax = fig.add_axes([xn, yn, wn, hn])
            ax.hist(samples_j[:, i], bins=xbins, alpha=alpha,
                    density=True, color='C' + str(j))
            ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 3))
            axes[ai, aj].spines["top"].set_visible(False)
            axes[ai, aj].spines["left"].set_visible(False)
            axes[ai, aj].spines["right"].set_visible(False)
            axes[ai, aj].spines["bottom"].set_visible(False)
            axes[ai, aj].set_xticks([])
            axes[ai, aj].set_yticks([])
            axes[ai, aj].set_xticklabels([])
            axes[ai, aj].set_yticklabels([])
        else:
            axes[ai, aj].hist(samples_j[:, i], bins=xbins, alpha=alpha,
                    density=True, label=discrepancy_names[j],
                    color='C' + str(j))
axes[0, 0].legend(loc='lower left', bbox_to_anchor=(0, 1.1), ncol=4,
        bbox_transform=axes[0, 0].transAxes)
plt.savefig('%s/%s' % (savedir, saveas), dpi=200, bbox_inches='tight')
plt.close('all')

