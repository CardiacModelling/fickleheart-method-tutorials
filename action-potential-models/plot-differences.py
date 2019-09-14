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

# Get voltage traces (stim1hz)
stim_seq = p.stim1hz
times = p.stim1hz_times
idx_until = len(times) // 5
stim_seq = stim_seq[:idx_until]
times = times[:idx_until]

model_tnnp = m.Model('./mmt-model-files/tnnp-2004.mmt', stim_seq=stim_seq)
model_tnnp.set_name('tnnp-2004')

model_fink = m.Model('./mmt-model-files/fink-2008.mmt', stim_seq=stim_seq)
model_fink.set_name('fink-2008')

# Simulate and add noise
voltage = model_tnnp.simulate(np.ones(model_tnnp.n_parameters()), times)

# Simulate current
currents_tnnp = model_tnnp.current(np.ones(model_tnnp.n_parameters()),
        voltage, times)
currents_fink = model_fink.current(np.ones(model_fink.n_parameters()),
        voltage, times)

# Plot
fig = plt.figure(figsize=(12, 5))
grid = plt.GridSpec(5, 2, hspace=0.1, wspace=0.225)
# Vm
for i in range(2):
    vm_ax = fig.add_subplot(grid[0, i])
    vm_ax.plot(times, voltage, 'k', lw=2)
    vm_ax.set_ylabel(r'$\mathregular{V_m}$ (mV)', fontsize=14)
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
for i, c in enumerate(m.parameters):
    n = c[:-2]
    gx, gy = i % 4, int(i // 4)
    ax = fig.add_subplot(grid[gx + 1, gy])
    ax.plot(times, currents_tnnp[n], c='#2b8cbe', lw=2)
    ax.fill_between(times, 0, currents_tnnp[n], color='#a6bddb', alpha=0.5)
    ax.plot(times, currents_fink[n], c='#2ca25f', lw=2)
    ax.fill_between(times, 0, currents_fink[n], color='#a6dbbd', alpha=0.5)
    # Change y-ticks
    yticks = ax.get_yticks()
    ax.set_yticks([yticks[1], yticks[-2]])
    ax.set_ylabel(currents[i][0], rotation=0, fontsize=16)
    ax.get_yaxis().set_label_coords(-0.15, 0.5)
    # Change frame and x-ticks
    if gx + 1 != 4:
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
        ax.set_xlabel(r'Time (ms)', fontsize=14)
        # frame off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

grid.tight_layout(fig, pad=1.0, rect=(0.01, 0.01, 1, 1))
plt.savefig('fig/model-differences.png', bbox_inch='tight', pad_inches=0)
#plt.savefig('fig/model-differences.pdf', format='pdf', bbox_inch='tight',
#        pad_inches=0, transparent=True)
