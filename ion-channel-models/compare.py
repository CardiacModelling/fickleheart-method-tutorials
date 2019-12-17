#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('./method')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import model as m

"""
Run fit.
"""

predict_list = ['sinewave', 'staircase', 'activation', 'ap']

try:
    which_predict = sys.argv[1]
except:
    print('Usage: python %s [str:which_predict]' % os.path.basename(__file__))
    sys.exit()

if which_predict not in predict_list:
    raise ValueError('Input data %s is not available in the predict list' \
            % which_predict)

# Get all input variables
import importlib
sys.path.append('./mmt-model-files')
info_id_a = 'model_A'
info_a = importlib.import_module(info_id_a)
info_id_b = 'model_B'
info_b = importlib.import_module(info_id_b)

data_dir = './data'

savedir = './fig/compare'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

data_file_name = 'data-%s.csv' % which_predict
print('Predicting ', data_file_name)
saveas = 'compare-sinewave-' + which_predict

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

# Model
model_a = m.Model(info_a.model_file,
        variables=info_a.parameters,
        current_readout=info_a.current_list,
        set_ion=info_a.ions_conc,
        transform=None,
        temperature=273.15 + info_a.temperature,  # K
        )
model_b = m.Model(info_b.model_file,
        variables=info_b.parameters,
        current_readout=info_b.current_list,
        set_ion=info_b.ions_conc,
        transform=None,
        temperature=273.15 + info_b.temperature,  # K
        )

# Update protocol
model_a.set_fixed_form_voltage_protocol(protocol, protocol_times)
model_b.set_fixed_form_voltage_protocol(protocol, protocol_times)

# Load calibrated parameters
load_seed = 542811797
fix_idx = [1]
calloaddir_a = './out/' + info_id_a
calloaddir_b = './out/' + info_id_b
cal_params_a = []
cal_params_b = []
for i in fix_idx:
    cal_params_a.append(np.loadtxt('%s/%s-solution-%s-%s.txt' % \
            (calloaddir_a, 'sinewave', load_seed, i)))
    cal_params_b.append(np.loadtxt('%s/%s-solution-%s-%s.txt' % \
            (calloaddir_b, 'sinewave', load_seed, i)))

# Predict
predictions_a = []
for p in cal_params_a:
    predictions_a.append(model_a.simulate(p, times))
predictions_b = []
for p in cal_params_b:
    predictions_b.append(model_b.simulate(p, times))

# Plot
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 3.5),
        gridspec_kw={'height_ratios': [1, 2.5]})
is_predict = ' prediction' if which_predict != 'sinewave' else ''
sim_protocol = model_a.voltage(times)  # model_b should give the same thing
axes[0].plot(times, sim_protocol, c='#7f7f7f')
axes[0].set_ylabel('Voltage\n(mV)', fontsize=16)
axes[1].plot(times, data, alpha=0.5, label='Data')
for i, p in zip(fix_idx, predictions_a):
    axes[1].plot(times, p, label='Model A' + is_predict)
for i, p in zip(fix_idx, predictions_b):
    axes[1].plot(times, p, label='Model B' + is_predict)

# Zooms
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
sys.path.append('./protocol-time-series')
zoom = importlib.import_module(which_predict + '_to_zoom')
axes[1].set_ylim(zoom.set_ylim)
for i_zoom, (w, h, loc) in enumerate(zoom.inset_setup):
    axins = inset_axes(axes[1], width=w, height=h, loc=loc,
            axes_kwargs={"facecolor" : "#f0f0f0"})
    axins.plot(times, data, alpha=0.5)
    for i, p in zip(fix_idx, predictions_a):
        axins.plot(times, p)
    for i, p in zip(fix_idx, predictions_b):
        axins.plot(times, p)
    axins.set_xlim(zoom.set_xlim_ins[i_zoom])
    axins.set_ylim(zoom.set_ylim_ins[i_zoom])
    #axins.yaxis.get_major_locator().set_params(nbins=3)
    #axins.xaxis.get_major_locator().set_params(nbins=3)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    pp, p1, p2 = mark_inset(axes[1], axins, loc1=zoom.mark_setup[i_zoom][0],
            loc2=zoom.mark_setup[i_zoom][1], fc="none", lw=0.75, ec='k')
    pp.set_fill(True); pp.set_facecolor("#f0f0f0")

if which_predict == 'sinewave':
    axes[1].legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=3,
            bbox_transform=axes[0].transAxes)
axes[1].set_ylabel('Current (pA)', fontsize=16)
axes[1].set_xlabel('Time (ms)', fontsize=16)
for i in range(2):
    axes[i].set_xlim((times[0], times[-1]))

plt.subplots_adjust(hspace=0)
plt.savefig('%s/%s' % (savedir, saveas), bbox_inches='tight', dpi=200)
plt.close()

