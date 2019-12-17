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
from scipy.interpolate import interp1d
from scipy.stats import norm as scipy_stats_norm

"""
Posterior predictives with different discrepancy models.

This script plots the cached posterior predictives generated by `posterior.py`,
`posterior-gp.py`, and `posterior-arma.py`.
"""

model_list = ['A', 'B', 'C']
predict_list = ['sinewave', 'staircase', 'ap']
discrepancy_list = ['', '-gp', '-gp-ov', '-arma_2_2']
load_list = ['-iid', '-gp', '-gp', '-armax']
discrepancy_names = ['iid noise', 'GP(t)', 'GP(O, V)', 'ARMA(2, 2)']

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

info_id = 'model_%s' % which_model
savedir = './fig/compare'
if not os.path.isdir(savedir):
    os.makedirs(savedir)
saveas = 'compare-' + info_id + '-sinewave-%s-pp' % which_predict

if which_predict == 'sinewave':
    zoom = [-1250, 800]
elif which_predict == 'staircase':
    zoom = [-600, 1600]
elif which_predict == 'ap':
    zoom = [-200, 4200]

data_dir = './data'
data_file_name = 'data-%s.csv' % which_predict
print('Predicting ', data_file_name)

# Load data
data = np.loadtxt(data_dir + '/' + data_file_name,
                  delimiter=',', skiprows=1)  # headers
times = data[:, 0]
data = data[:, 1]

# Load protocol
protocol = np.loadtxt('./protocol-time-series/%s.csv' % which_predict,
        skiprows=1, delimiter=',')
protocol_times = protocol[:, 0]
protocol = protocol[:, 1]
voltage = interp1d(protocol_times, protocol, kind='linear')(times)

# Load cached posterior prediction
times_list = []
ppc_mean_list = []
ppc_sd_list = []
ppc_model_mean_list = []
ppc_model_sd_list = []
ppc_disc_mean_list = []
ppc_disc_sd_list = []
for i, (d, l) in enumerate(zip(discrepancy_list, load_list)):
    loaddir = './fig/mcmc-' + info_id + d + '/raw'
    loadas = info_id + '-sinewave-' + which_predict
    times_list.append(np.loadtxt('%s/%s-pp-time.txt' % (loaddir, loadas)))

    ppc_mean_list.append(np.loadtxt('%s/%s-pp%s-mean.txt'
            % (loaddir, loadas, l)))
    ppc_sd_list.append(np.loadtxt('%s/%s-pp%s-sd.txt' % (loaddir, loadas, l)))

    ppc_model_mean_list.append(np.loadtxt('%s/%s-pp-only-model-mean.txt'
            % (loaddir, loadas)))
    ppc_model_sd_list.append(np.loadtxt('%s/%s-pp-only-model-sd.txt'
            % (loaddir, loadas)))

    ppc_disc_mean_list.append(np.loadtxt('%s/%s-pp-only%s-mean.txt'
            % (loaddir, loadas, l)))
    ppc_disc_sd_list.append(np.loadtxt('%s/%s-pp-only%s-sd.txt'
            % (loaddir, loadas, l)))

n_sd = scipy_stats_norm.ppf(1. - .05 / 2.)

# Plot model + discrepancy
if (which_model == 'A') and (which_predict in ['sinewave', 'staircase']):
    fig, axes = plt.subplots(len(discrepancy_list) + 1, 1, sharex=True,
            figsize=(8, 5),
            gridspec_kw={'height_ratios': [1] + [2] * len(discrepancy_list)})
    axes[0].plot(times, voltage, c='#7f7f7f')
    axes[0].set_ylabel('Voltage\n(mV)')
    axes[0].set_title('ODE model with discrepancy', loc='left')
    axes[0].set_xlim((times[0], times[-1]))
    for i, d in enumerate(discrepancy_names):
        ppc_mean = ppc_mean_list[i]
        ppc_sd = ppc_sd_list[i]
        a = 0.5 #- i * 0.25
        axes[i + 1].plot(times, data, alpha=0.5, c='#7f7f7f', label='Data')
        axes[i + 1].plot(times, ppc_mean, c='C' + str(i), alpha=0.9, lw=0.5,
                label=d)# + ' mean')
        axes[i + 1].fill_between(times,
                ppc_mean - n_sd * ppc_sd,
                ppc_mean + n_sd * ppc_sd,
                facecolor='C' + str(i), linewidth=0, alpha=a,)
                #label=d + ' 95% C.I.')
        if which_predict in ['sinewave']:
            axes[i + 1].legend(loc=3)
        elif which_predict in ['staircase']:
            axes[i + 1].legend(loc=2, ncol=2)
        axes[i + 1].set_ylabel('Current\n(pA)')
        axes[i + 1].set_ylim(zoom)
        axes[i + 1].set_xlim((times[0], times[-1]))
    # Add arrows...
    if which_predict == 'staircase':
        for i in range(1, 5):
            axes[i].annotate("", xy=(2500, 200), xytext=(3250, 600),
                    arrowprops=dict(arrowstyle="->", color='#cb181d'))
        axes[2].annotate("", xy=(7550, 500), xytext=(8300, 900),
                arrowprops=dict(arrowstyle="->", color='#0570b0'))
    axes[-1].set_xlabel('Time (ms)')
    plt.subplots_adjust(hspace=0)
    plt.savefig('%s/%s' % (savedir, saveas), dpi=200, bbox_inches='tight')
    plt.close()
else:
    fig, axes = plt.subplots(len(discrepancy_list) + 1, 1, sharex=True,
            figsize=(8, 8),
            gridspec_kw={'height_ratios': [1] + [2] * len(discrepancy_list)})
    axes[0].plot(times, voltage, c='#7f7f7f')
    axes[0].set_ylabel('Voltage (mV)')
    axes[0].set_title('ODE model with discrepancy', loc='left')
    for i, d in enumerate(discrepancy_names):
        ppc_mean = ppc_mean_list[i]
        ppc_sd = ppc_sd_list[i]
        a = 0.5 #- i * 0.25
        axes[i + 1].plot(times, data, alpha=0.5, c='#7f7f7f', label='Data')
        axes[i + 1].plot(times, ppc_mean, c='C' + str(i), alpha=0.9, lw=0.5,
                label=d + ' mean')
        axes[i + 1].fill_between(times,
                ppc_mean - n_sd * ppc_sd,
                ppc_mean + n_sd * ppc_sd,
                facecolor='C' + str(i), linewidth=0, alpha=a,
                label=d + ' 95% C.I.')
        axes[i + 1].legend()
        axes[i + 1].set_ylabel('Current (pA)')
        axes[i + 1].set_ylim(zoom)
    axes[-1].set_xlabel('Time (ms)')
    plt.subplots_adjust(hspace=0)
    plt.savefig('%s/%s' % (savedir, saveas), dpi=200, bbox_inches='tight')
    plt.close()

# Plot model only
fig, axes = plt.subplots(len(discrepancy_list) + 1, 1, sharex=True,
        figsize=(8, 8),
        gridspec_kw={'height_ratios': [1] + [2] * len(discrepancy_list)})
axes[0].plot(times, voltage, c='#7f7f7f')
axes[0].set_ylabel('Voltage (mV)')
axes[0].set_title('ODE model only', loc='left')
for i, d in enumerate(discrepancy_names):
    ppc_mean = ppc_model_mean_list[i]
    ppc_sd = ppc_model_sd_list[i]
    a = 0.5 #- i * 0.25
    axes[i + 1].plot(times, data, alpha=0.5, c='#7f7f7f', label='Data')
    axes[i + 1].plot(times, ppc_mean, c='C' + str(i), alpha=0.9, lw=0.5,
            label=d + ' mean')
    axes[i + 1].fill_between(times,
            ppc_mean - n_sd * ppc_sd,
            ppc_mean + n_sd * ppc_sd,
            facecolor='C' + str(i), linewidth=0, alpha=a,
            label=d + ' 95% C.I.')
    axes[i + 1].legend()
    axes[i + 1].set_ylabel('Current (pA)')
    axes[i + 1].set_ylim(zoom)
axes[-1].set_xlabel('Time (ms)')
plt.subplots_adjust(hspace=0)
plt.savefig('%s/%s-only-model' % (savedir, saveas), dpi=200,
        bbox_inches='tight')
plt.close()

# Plot discrepancy only
fig, axes = plt.subplots(len(discrepancy_list) + 1, 1, sharex=True,
        figsize=(8, 8),
        gridspec_kw={'height_ratios': [1] + [2] * len(discrepancy_list)})
axes[0].plot(times, voltage, c='#7f7f7f')
axes[0].set_ylabel('Voltage (mV)')
axes[0].set_title('Discrepancy only', loc='left')
for i, d in enumerate(discrepancy_names):
    ppc_mean = ppc_disc_mean_list[i]
    ppc_sd = ppc_disc_sd_list[i]
    a = 0.5 #- i * 0.25
    axes[i + 1].plot(times, data - ppc_model_mean_list[i], alpha=0.5,
            c='#7f7f7f', label='Data - ODE model')
    axes[i + 1].plot(times, ppc_mean, c='C' + str(i), alpha=0.9, lw=0.5,
            label=d + ' mean')
    axes[i + 1].fill_between(times,
            ppc_mean - n_sd * ppc_sd,
            ppc_mean + n_sd * ppc_sd,
            facecolor='C' + str(i), linewidth=0, alpha=a,
            label=d + ' 95% C.I.')
    axes[i + 1].legend()
    axes[i + 1].set_ylabel('Current (pA)')
    #axes[i + 1].set_ylim(zoom)
axes[-1].set_xlabel('Time (ms)')
plt.subplots_adjust(hspace=0)
plt.savefig('%s/%s-only-disc' % (savedir, saveas), dpi=200,
        bbox_inches='tight')
plt.close()
