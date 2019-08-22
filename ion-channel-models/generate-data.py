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

savedir = './data'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

iid_noise_sigma = 25  # pA

np.random.seed(101)

# Get all input variables
import importlib
sys.path.append('./mmt-model-files')
info_idc = 'model_C'
infoc = importlib.import_module(info_idc)

modelc = m.Model(infoc.model_file,
        variables=infoc.parameters,
        current_readout=infoc.current_list,
        set_ion=infoc.ions_conc,
        temperature=273.15 + infoc.temperature,  # K
        )

#
# Activation step protocol
#
# Load protocol
protocol = [-80, 200, 20, 500, -40, 500, -80, 200]
dt = 0.1
times = np.arange(0, np.sum(protocol[1::2]), dt)

# Update protocol
modelc.set_voltage_protocol(protocol)

# Simulate and add noise
simulated_data = modelc.simulate(infoc.base_param, times)
simulated_data += np.random.normal(0, iid_noise_sigma,
        size=simulated_data.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.subplot(211)
plt.plot(times, modelc.voltage(times), c='#7f7f7f')
plt.ylabel('Voltage (mV)')

plt.subplot(212)
plt.plot(times, simulated_data, label='Model C')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Current (pA)')

plt.subplots_adjust(hspace=0)
plt.savefig('%s/data-activation.png' % (savedir), bbox_inches='tight')
plt.close()

# Save
np.savetxt('%s/data-activation.csv' % (savedir),
        np.array([times, simulated_data]).T, delimiter=',', comments='',
        header='\"time\",\"current\"')

#
# Sine wave protocol
#
# Load protocol
protocol = np.loadtxt('./protocol-time-series/sinewave.csv', skiprows=1,
        delimiter=',')
times = protocol[:, 0]
protocol = protocol[:, 1]

# Update protocol
modelc.set_fixed_form_voltage_protocol(protocol, times)

# Simulate and add noise
simulated_data = modelc.simulate(infoc.base_param, times)
simulated_data += np.random.normal(0, iid_noise_sigma,
        size=simulated_data.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.subplot(211)
plt.plot(times, modelc.voltage(times), c='#7f7f7f')
plt.ylabel('Voltage (mV)')

plt.subplot(212)
plt.plot(times, simulated_data, label='Model C')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Current (pA)')

plt.subplots_adjust(hspace=0)
plt.savefig('%s/data-sinewave.png' % (savedir), bbox_inches='tight')
plt.close()

# Save
np.savetxt('%s/data-sinewave.csv' % (savedir),
        np.array([times, simulated_data]).T, delimiter=',', comments='',
        header='\"time\",\"current\"')

#
# APs protocol
#
# Load protocol
protocol = np.loadtxt('./protocol-time-series/ap.csv', skiprows=1,
        delimiter=',')
times = protocol[:, 0]
protocol = protocol[:, 1]

# Update protocol
modelc.set_fixed_form_voltage_protocol(protocol, times)

# Simulate and add noise
simulated_data = modelc.simulate(infoc.base_param, times)
simulated_data += np.random.normal(0, iid_noise_sigma,
        size=simulated_data.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.subplot(211)
plt.plot(times, modelc.voltage(times), c='#7f7f7f')
plt.ylabel('Voltage (mV)')

plt.subplot(212)
plt.plot(times, simulated_data, label='Model C')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Current (pA)')

plt.subplots_adjust(hspace=0)
plt.savefig('%s/data-ap.png' % (savedir), bbox_inches='tight')
plt.close()

# Save
np.savetxt('%s/data-ap.csv' % (savedir),
        np.array([times, simulated_data]).T, delimiter=',', comments='',
        header='\"time\",\"current\"')

#
# Staircase protocol
#
# Load protocol
protocol = np.loadtxt('./protocol-time-series/staircase.csv', skiprows=1,
        delimiter=',')
times = protocol[:, 0]
protocol = protocol[:, 1]

# Update protocol
modelc.set_fixed_form_voltage_protocol(protocol, times)

# Simulate and add noise
simulated_data = modelc.simulate(infoc.base_param, times)
simulated_data += np.random.normal(0, iid_noise_sigma,
        size=simulated_data.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.subplot(211)
plt.plot(times, modelc.voltage(times), c='#7f7f7f')
plt.ylabel('Voltage (mV)')

plt.subplot(212)
plt.plot(times, simulated_data, label='Model C')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Current (pA)')

plt.subplots_adjust(hspace=0)
plt.savefig('%s/data-staircase.png' % (savedir), bbox_inches='tight')
plt.close()

# Save
np.savetxt('%s/data-staircase.csv' % (savedir),
        np.array([times, simulated_data]).T, delimiter=',', comments='',
        header='\"time\",\"current\"')

