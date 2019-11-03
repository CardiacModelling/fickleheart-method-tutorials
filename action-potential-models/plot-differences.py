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

import model as m
import protocol as p

# Current names
currents = [(r'$\mathregular{I_{Na}}$', '#2b8cbe', '#a6bddb'),
         (r'$\mathregular{I_{CaL}}$', '#2b8cbe', '#a6bddb'),
         (r'$\mathregular{I_{Kr}}$', '#2b8cbe', '#a6bddb'),
         (r'$\mathregular{I_{Ks}}$', '#2b8cbe', '#a6bddb'),
         (r'$\mathregular{I_{to}}$', '#2b8cbe', '#a6bddb'),
         (r'$\mathregular{I_{NaCa}}$', '#2ca25f', '#a6dbbd'),
         (r'$\mathregular{I_{K1}}$', '#2b8cbe', '#a6bddb'),
         (r'$\mathregular{I_{NaK}}$', '#54278f', '#756bb1'),
         ]

take_out = [0, 5, 7]  # identical kinetics

# Get voltage traces (stim1hz)
stim_seq = p.stim1hz
times = p.stim1hz_times
idx_until = int(len(times) // 10)
stim_seq = stim_seq[:idx_until]
times = times[:idx_until]

model_tnnp = m.Model('./mmt-model-files/tnnp-2004.mmt', stim_seq=stim_seq)
model_tnnp.set_name('tnnp-2004')

model_fink = m.Model('./mmt-model-files/fink-2008.mmt', stim_seq=stim_seq)
model_fink.set_name('fink-2008')

# Simulate voltage
voltage = model_tnnp.simulate(np.ones(model_tnnp.n_parameters()), times)

# Re-noramlisation factor
renormalisation = np.asarray(model_fink.original) \
        / np.asarray(model_tnnp.original)

# Just double checking, should all be the same
renormalised_fink = np.ones(model_fink.n_parameters()) #* renormalisation

# Simulate current
currents_tnnp = model_tnnp.current(np.ones(model_tnnp.n_parameters()),
        voltage, times)
currents_fink = model_fink.current(renormalised_fink, voltage, times)

# Plot
fig = plt.figure(figsize=(12, 6))
grid = plt.GridSpec(4, 2, hspace=0.1, wspace=0.225)
# Vm
for i in range(2):
    vm_ax = fig.add_subplot(grid[0, i])
    vm_ax.plot(times, voltage, 'k', lw=2)
    vm_ax.set_ylabel(r'$\mathregular{V_m}$ (mV)', fontsize=16)
    vm_ax.tick_params(axis='x',
                   which='both',
                   bottom=False,
                   top=False,
                   labelbottom=False
            )
    # frame off
    vm_ax.spines['top'].set_visible(False)
    vm_ax.spines['right'].set_visible(False)
    vm_ax.spines['bottom'].set_visible(False)

# currents
plot_parameters = [p for i, p in enumerate(m.parameters) if i not in take_out]
plot_currents = [c for i, c in enumerate(currents) if i not in take_out]
for i, c in enumerate(plot_parameters):
    n = c[:-2]
    gx, gy = i % 3, int(i // 3)
    ax = fig.add_subplot(grid[gx + 1, gy])
    ax.plot(times, currents_tnnp[n], c='#2b8cbe', lw=2, label='Model T')
    ax.fill_between(times, 0, currents_tnnp[n], color='#a6bddb', alpha=0.5)
    ax.plot(times, currents_fink[n], c='#2ca25f', lw=2, label='Model F')
    ax.fill_between(times, 0, currents_fink[n], color='#a6dbbd', alpha=0.5)
    if i == 0:
        ax.legend(bbox_to_anchor=(0.6, 0.9), frameon=False, fontsize=13)
    # Change y-ticks
    yticks = ax.get_yticks()
    ax.set_yticks([yticks[1], yticks[-2]])
    ax.set_ylabel(plot_currents[i][0] + '\n(A/F)', rotation=0, fontsize=16)
    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    # Change frame and x-ticks
    if i not in [2, 4]:
        ax.tick_params(axis='x',
                       which='both',
                       bottom=False,
                       top=False,
                       labelbottom=False
                )
        # frame off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    else:
        ax.tick_params(axis='x',
                       which='major',
                       size=8,
                       labelsize=14)
        ax.set_xlabel(r'Time (ms)', fontsize=16)
        # frame off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Zoom in for Ito
    if c == 'ito.s':
        axins = inset_axes(ax, width=2.5, height=0.8, loc='center right',
                axes_kwargs={"facecolor" : "#f0f0f0"})
        axins.plot(times, currents_tnnp[n], c='#2b8cbe', lw=2)
        axins.fill_between(times, 0, currents_tnnp[n], color='#a6bddb',
                alpha=0.5)
        axins.plot(times, currents_fink[n], c='#2ca25f', lw=2)
        axins.fill_between(times, 0, currents_fink[n], color='#a6dbbd',
                alpha=0.5)
        axins.set_xlim([45, 100])
        axins.set_ylim([-0.2, 10.5])
        #axins.yaxis.get_major_locator().set_params(nbins=3)
        #axins.xaxis.get_major_locator().set_params(nbins=3)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        pp, p1, p2 = mark_inset(ax, axins, loc1=1, loc2=4, fc="none", lw=0.75,
                ec='k')
        pp.set_fill(True); pp.set_facecolor("#f0f0f0")

grid.tight_layout(fig, pad=1.0, rect=(0.01, 0.01, 1, 1))
grid.update(wspace=0.175, hspace=0.1)
plt.savefig('fig/model-differences.png', bbox_inch='tight', pad_inches=0)
plt.savefig('fig/model-differences.pdf', format='pdf', bbox_inch='tight',
        pad_inches=0)
