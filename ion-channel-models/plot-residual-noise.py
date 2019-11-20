#!/usr/bin/env python3
import sys
sys.path.append('method')
import os
import pints.plot
import numpy as np
import matplotlib.pyplot as plt
import model as m

import importlib
sys.path.append('./mmt-model-files')
infoa = importlib.import_module('model_A')
infob = importlib.import_module('model_B')
infoc = importlib.import_module('model_C')

savedir = './fig/residual'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Protocol
protocol = np.loadtxt('./protocol-time-series/sinewave.csv', skiprows=1,
        delimiter=',')
protocol_times = protocol[:, 0]
protocol = protocol[:, 1]

temperature = 23.0 + 273.15  # K

# Model
modela = m.Model(infoa.model_file,
        variables=infoa.parameters,
        current_readout=infoa.current_list,
        set_ion=infoa.ions_conc,
        transform=None,
        temperature=273.15 + infoa.temperature,  # K
        )
modelb = m.Model(infob.model_file,
        variables=infob.parameters,
        current_readout=infob.current_list,
        set_ion=infob.ions_conc,
        transform=None,
        temperature=273.15 + infob.temperature,  # K
        )
modelc = m.Model(infoc.model_file,
        variables=infoc.parameters,
        current_readout=infoc.current_list,
        set_ion=infoc.ions_conc,
        transform=None,
        temperature=273.15 + infoc.temperature,  # K
        )

# Update protocol
modela.set_fixed_form_voltage_protocol(protocol, protocol_times)
modelb.set_fixed_form_voltage_protocol(protocol, protocol_times)
modelc.set_fixed_form_voltage_protocol(protocol, protocol_times)

# Load data
data_dir = './data'
data_file_name = 'data-sinewave.csv'
data = np.loadtxt(data_dir + '/' + data_file_name,
                  delimiter=',', skiprows=1)  # headers
times = data[:, 0]
data = data[:, 1]

pa = np.loadtxt('out/model_A/sinewave-solution-542811797-1.txt')
pb = np.loadtxt('out/model_B/sinewave-solution-542811797-1.txt')
pc = np.copy(infoc.base_param)

sa = modela.simulate(pa, times)
sb = modelb.simulate(pb, times)
sc = modelc.simulate(pc, times)

ra = data - sa
rb = data - sb
rc = data - sc

# Fig 1
plt.figure(figsize=(8, 3))
plt.plot(times, ra, alpha=0.5, label='Model A residual')
plt.plot(times, rb, alpha=0.5, label='Model B residual')
plt.plot(times, rc, alpha=0.5, label='i.i.d. noise')
plt.legend()
plt.xlabel(r'Time (ms)')
plt.ylabel('Current (pA)')
plt.savefig('%s/residual-trace' % savedir, dpi=200)
plt.close()

# Fig 2
nbins = 200
plt.hist(ra, bins=nbins, alpha=0.5, label='Model A residual')
plt.hist(rb, bins=nbins, alpha=0.5, label='Model B residual')
plt.hist(rc, bins=nbins, alpha=0.5, label='i.i.d. noise')
plt.legend()
plt.xlabel('Current (pA)')
plt.ylabel('Frequency')
plt.savefig('%s/residual-hist' % savedir, dpi=200)
plt.close()

# Fig 3
def autocorrelation(samples, ax, legend='', max_lags=100):
    ax.acorr(samples - np.mean(samples), maxlags=max_lags)
    ax.set_xlim(-0.5, max_lags + 0.5)
    ax.legend([legend], loc='upper right')

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 4))
autocorrelation(ra, axes[0], legend='Model A residual')
autocorrelation(rb, axes[1], legend='Model B residual')
autocorrelation(rc, axes[2], legend='i.i.d. noise')

# Add x-label to final plot only
axes[-1].set_xlabel('Lag')
# Add vertical y-label to middle plot
axes[1].set_ylabel('Autocorrelation')
plt.tight_layout()
plt.savefig('%s/residual-autocorr' % savedir, dpi=200)
plt.close()

